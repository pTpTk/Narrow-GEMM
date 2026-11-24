#pragma once
#include "03_async.cuh"

namespace narrow {

__global__ void reduce_split_k(const float* partial, float* c, int split) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.f;
  for (int s = 0; s < split; ++s) sum += partial[size_t(s) * ElementsC + i];
  c[i] = sum;
}

inline void launch_reduce(const float* partial, float* c, int split,
                          cudaStream_t stream) {
  reduce_split_k<<<ElementsC / 256, 256, 0, stream>>>(partial, c, split);
}

template <int BN>
inline void launch_split_k(const bf16* a, const bf16* b, float* c,
                           float* partial, int split, cudaStream_t stream) {
  // Each partial is overwritten, so there is no timed memset or atomicAdd.
  launch_async<BN>(a, b, split == 1 ? c : partial, split, stream);
  if (split > 1) launch_reduce(partial, c, split, stream);
}

}  // namespace narrow
