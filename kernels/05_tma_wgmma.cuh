#pragma once
#include "wgmma.cuh"
#include "04_split_k.cuh"

namespace narrow {

template <int BN>
__global__ __launch_bounds__(128) void tma_serial_gemm(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b, float* out, int split) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& smem = *reinterpret_cast<HopperStorage<BN, 1>*>(storage);
  if (threadIdx.x == 0) {
    barrier_init(&smem.full[0], 1);
    barrier_init_fence();
  }
  __syncthreads();
  const int n_tile = blockIdx.x % (N / BN);
  const int slice = blockIdx.x / (N / BN);
  const int first = slice * (K / split);
  WgmmaAccumulators<BN> acc;
  for (int t = 0; t < (K / split) / HopperBK; ++t) {
    if (threadIdx.x == 0) {
      barrier_expect(&smem.full[0], (BM + BN) * HopperBK * sizeof(bf16));
      tma_load_2d(smem.a[0], &map_a, first + t * HopperBK, 0, &smem.full[0]);
      tma_load_2d(smem.b[0], &map_b, first + t * HopperBK, n_tile * BN, &smem.full[0]);
    }
    barrier_wait(&smem.full[0], t & 1);
    acc.consume(smem.a[0], smem.b[0], t == 0);
    // WGMMA wait covers this warpgroup. Synchronize before thread 0 can
    // issue the next pair of TMA loads into the same shared-memory slot.
    __syncthreads();
  }
  acc.store(out + size_t(slice) * ElementsC + n_tile * BN);
}

template <int BN>
inline void configure_tma_serial() {
  CUDA_CHECK(cudaFuncSetAttribute(tma_serial_gemm<BN>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(HopperStorage<BN, 1>)));
}

template <int BN>
inline void launch_tma_serial(CUtensorMap a, CUtensorMap b, float* out,
                              int split, cudaStream_t stream) {
  tma_serial_gemm<BN><<<(N / BN) * split, 128,
      sizeof(HopperStorage<BN, 1>), stream>>>(a, b, out, split);
}

}  // namespace narrow
