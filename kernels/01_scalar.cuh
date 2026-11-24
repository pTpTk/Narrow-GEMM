#pragma once
#include "common.cuh"

namespace narrow {

__global__ void scalar_gemm(const bf16* a, const bf16* b, float* c) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  float sum = 0.f;
  for (int k = 0; k < K; ++k)
    sum = fmaf(__bfloat162float(a[m * K + k]),
               __bfloat162float(b[n * K + k]), sum);
  c[m * N + n] = sum;
}

inline void launch_scalar(const bf16* a, const bf16* b, float* c,
                          cudaStream_t stream) {
  scalar_gemm<<<dim3(N / 256, M), 256, 0, stream>>>(a, b, c);
}

}  // namespace narrow
