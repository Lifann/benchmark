#pragma once

#include <stdlib.h>
#include <unistd.h>

#include <vector>

#include "cuda_runtime.h"

using namespace std;

#define CUDA_CHECK(err) \
  if (err != cudaSuccess) { \
    const char* msg = cudaGetErrorString(err); \
    fprintf(stderr, "CUDA_CHECK %s failed, error msg: %s", #err, msg); \
    exit(1); \
  }

namespace benchmark {

//template <typename T>
//void GenerateRandomMatrix(Matrix<T>* m, Shape shape, Stream stream) {
//  m->data();
//}

}  // namespace benchmark
