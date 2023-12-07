#include <stdio.h>
#include <memory>
#include <cmath>

#include "cuda_runtime.h"
#include "mma.h"
#include "crt/mma.h"
using namespace nvcuda;

#include "types.h"
#include "utils.h"

using namespace std;
using Shape = benchmark::Shape;
template <typename T> using Matrix = benchmark::Matrix<T>;
using fp32 = float;

constexpr int kTensorCoreDim = 16;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

template <typename T, typename T2>
__global__ void single_mma_matmul(const T* matrix_a,
    const T* matrix_b, T* output, int m,
    int n, int k) {
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T2, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T2, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(a_frag, matrix_a, k);
  wmma::load_matrix_sync(b_frag, matrix_b, n);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(output, c_frag, n, wmma::mem_row_major);
}

template <typename T>
void host_matmul_row_major(const T* matrix_a,
    const T* matrix_b, T* output, int M,
    int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T acc = 0.0;
      for (int k = 0; k < K; k++) {
        acc += matrix_a[i * K + k] * matrix_b[k * N + j];
      }
      output[i * N + j] = acc;
    }
  }
}

void test_matmal_bias_relu(int m, int n, int k) {
  Shape SA({m, k});
  Shape SB({n, k});
  Shape SC({m, n});
  Shape SD({m, n});

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream))

  unique_ptr<Matrix<fp32>> h_MA = Matrix<fp32>::Random(SA, /*on_device=*/false);
  unique_ptr<Matrix<fp32>> h_MB = Matrix<fp32>::Random(SB, /*on_device=*/false);
  unique_ptr<Matrix<fp32>> h_MD = Matrix<fp32>::Zeros(SD, /*on_device=*/false);

  //unique_ptr<Matrix<fp32>> MA = Matrix<fp32>::Random(SA, /*on_device=*/true);
  //unique_ptr<Matrix<fp32>> MB = Matrix<fp32>::Random(SB, /*on_device=*/true);
  //unique_ptr<Matrix<fp32>> MD = Matrix<fp32>::Zeros(SC, /*on_device=*/true);

  unique_ptr<Matrix<fp32>> MA(new Matrix<fp32>(SA, /*on_device=*/true));
  unique_ptr<Matrix<fp32>> MB(new Matrix<fp32>(SB, /*on_device=*/true));
  unique_ptr<Matrix<fp32>> MD(new Matrix<fp32>(SD, /*on_device=*/true));
  MA->CopyFrom(*(h_MA.get()), stream);
  MB->CopyFrom(*(h_MB.get()), stream);
  MD->CopyFrom(*(h_MD.get()), stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));


  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, /*device=*/0));


  int block_dim = device_prop.maxThreadsPerBlock;
  printf("max block_dim: %d\n", block_dim);
  int max_threads_per_sm = device_prop.maxThreadsPerMultiProcessor;
  printf("max_threads_per_sm: %d\n", max_threads_per_sm);
  int max_grid_size[3];
  for (int i = 0; i < 3; i++) {
    max_grid_size[i] = device_prop.maxGridSize[i];
    printf("max_grid_size[%d]: %d\n", i, max_grid_size[i]);
  }
  int multiProcessorCount = device_prop.multiProcessorCount;
  printf("multiProcessorCount: %d\n", multiProcessorCount);

  //int grid_size = multiProcessorCount;
  //int block_size = 128;  // For one of tfp32 block.

  host_matmul_row_major<fp32>(h_MA->data(), h_MB->data(), h_MD->data(), m, n, k);
  printf("----> check run mma\n");
  single_mma_matmul<fp32, wmma::precision::tf32><<<1, 32, 0, stream>>>(MA->data(), MB->data(), MD->data(), m, n, k);

  unique_ptr<Matrix<fp32>> h_recv(new Matrix<fp32>(SD, /*on_device=*/false));
  h_recv->CopyFrom(*(MD.get()), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  fp32 min_relative_diff = 1000.0f;
  fp32 max_relative_diff = 0.0f;
  for (int i = 0; i < SD.volumn(); i++) {
    fp32 diff = std::abs(h_recv->data()[i] - h_MD->data()[i]);
    if (diff > max_relative_diff) {
	  max_relative_diff = diff;
    }
    if (diff < min_relative_diff) {
	  min_relative_diff = diff;
    }
    printf("absolute diff = %f, relative diff = %f\n", diff, diff / h_MD->data()[i]);
  }
  printf("max_relative_diff: %f\n", max_relative_diff);
  printf("min_relative_diff: %f\n", min_relative_diff);

  CUDA_CHECK(cudaStreamDestroy(stream))
}

int main() {
  test_matmal_bias_relu(16, 16, 8);

  return 0;
}
