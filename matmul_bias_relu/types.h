#pragma once

#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <utility>
#include <memory>
#include <random>

#include "cuda_runtime.h"
#include "driver_types.h"

#include "utils.h"

using namespace std;

namespace benchmark {

using Stream = cudaStream_t;

class Shape {
 public:
  Shape(vector<int> dims) : dims_(dims) {}
  Shape(const Shape& other) {
    CopyFrom(other);
  }
  void operator=(const Shape& other) {
    CopyFrom(other);
  }
  bool operator != (const Shape& other) {
    if (dims_.size() != other.ndims()) {
      return true;
    }
    for (int i = 0; i < dims_.size(); i++) {
      if (dims_[i] != other.dim(i)) {
        return true;
      }
    }
    return false;
  }
  int ndims() const { return dims_.size(); }
  int dim(int i) const { return dims_[i]; }

  int volumn() const {
    int v = 1;
    for (size_t i = 0; i < dims_.size(); i++) {
      v *= dims_[i];
    }
    return v;
  }

 private:
  void CopyFrom(const Shape& other) {
    dims_.resize(other.ndims());
    for (int i = 0; i < other.ndims(); i++) {
      dims_[i] = other.dim(i);
    }
  }

 private:
  vector<int> dims_;
};

template <typename T>
class Matrix {
 public:
  Matrix(Shape shape, bool on_device) : shape_(shape), on_device_(on_device) {
    if (on_device_) {
      CUDA_CHECK(cudaMalloc(&data_, shape_.volumn() * sizeof(T)));
      CUDA_CHECK(cudaMemset(data_, 0, shape_.volumn() * sizeof(T)));
    } else {
      data_ = (T*) malloc(shape_.volumn() * sizeof(T));
      memset(data_, 0, shape_.volumn() * sizeof(T));
    }
  }

  ~Matrix() {
    if (data_) {
      if (on_device_) {
        CUDA_CHECK(cudaFree(data_));
      } else {
        free(data_);
      }
    }
    data_ = nullptr;
  }

  static unique_ptr<Matrix<T>> Random(Shape& shape, bool on_device) {
    unique_ptr<Matrix<T>> m(new Matrix<T>(shape, on_device));
    Matrix* host_m = new Matrix(shape, /*on_device=*/false);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0f, 1.0f);
    for (int i = 0; i < shape.volumn(); i++) {
      host_m->data()[i] = dist(gen);
    }
    if (!on_device) {
      m.reset(host_m);
    } else {
      CUDA_CHECK(cudaMemcpy(m->data(), host_m->data(), shape.volumn() * sizeof(T), cudaMemcpyHostToDevice));
      delete host_m;
    }
    return m;
  }

  static unique_ptr<Matrix<T>> Zeros(Shape& shape, bool on_device) {
    unique_ptr<Matrix<T>> m(new Matrix<T>(shape, on_device));
    return m;
  }

  bool CopyFrom(Matrix<T>& other, Stream stream) {
    if (shape_ != other.shape()) {
      return false;
    }
    cudaMemcpyKind kind;
    if (on_device_ && other.on_device()) {
      kind = cudaMemcpyDeviceToDevice;
    } else if (on_device_ && !other.on_device()) {
      kind = cudaMemcpyHostToDevice;
    } else if (!on_device_ && other.on_device()) {
      kind = cudaMemcpyDeviceToHost;
    } else if (!on_device_ && other.on_device()) {
      kind = cudaMemcpyHostToHost;
    }
    CUDA_CHECK(cudaMemcpyAsync(data_, other.data(), shape().volumn() * sizeof(T), kind, stream));
    return true;
  }

  T* data() { return data_; }
  bool on_device() const { return on_device_; }
  Shape shape() const { return shape_; }

 private:
  T* data_ = nullptr;
  Shape shape_;
  bool on_device_;
};

}  // namespace benchmark
