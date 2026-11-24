#pragma once
#include "common.cuh"
#include <cublas_v2.h>

// cuBLAS is used only by the benchmark reference, never by a custom kernel.
namespace narrow {

inline void blas_check(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s:%d: cuBLAS status %d\n", file, line, int(status));
    std::exit(EXIT_FAILURE);
  }
}
#define BLAS_CHECK(call) ::narrow::blas_check((call), __FILE__, __LINE__)

inline void cublas_gemm(cublasHandle_t handle, const bf16* a, const bf16* b,
                        float* c) {
  const float alpha = 1.f, beta = 0.f;
  // Row-major C is column-major C^T. Compute C^T = B^T A^T.
  // B is KxN column-major, while A's bytes are KxM column-major.
  BLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
      &alpha, b, CUDA_R_16BF, K, a, CUDA_R_16BF, K,
      &beta, c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

}  // namespace narrow
