#pragma once
#include "tile.cuh"

namespace narrow {

template <int BN>
__global__ void mma_gemm(const bf16* a, const bf16* b, float* c) {
  extern __shared__ __align__(32) unsigned char warp_storage[];
  auto& smem = *reinterpret_cast<Tiles<BN, 1>*>(warp_storage);
  Accumulators<BN> acc;
  acc.clear();
  b += blockIdx.x * BN * K;
  for (int k0 = 0; k0 < K; k0 += BK) {
    copy_rows<BM, false>(smem.a[0], a, k0);
    copy_rows<BN, false>(smem.b[0], b, k0);
    __syncthreads();  // Publish all vector loads.
    acc.mma(smem.a[0], smem.b[0]);
    __syncthreads();  // All warps must finish before shared memory is overwritten.
  }
  acc.store(c + blockIdx.x * BN);
}

template <int BN>
inline void configure_mma() {
  CUDA_CHECK(cudaFuncSetAttribute(mma_gemm<BN>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Tiles<BN, 1>)));
}

template <int BN>
inline void launch_mma(const bf16* a, const bf16* b, float* c, cudaStream_t stream) {
  mma_gemm<BN><<<N / BN, Threads, sizeof(Tiles<BN, 1>), stream>>>(a, b, c);
}

}  // namespace narrow
