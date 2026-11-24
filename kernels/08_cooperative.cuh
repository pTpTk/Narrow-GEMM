#pragma once
#include "06_warp_specialized.cuh"

namespace narrow {

// A single consumer warpgroup also issues future TMA loads. This avoids all
// dedicated producer threads and empty-slot arrivals. End-of-iteration CTA
// synchronization protects the slot that will be refilled next iteration.
template <int BN, int Stages, int BatchK>
__global__ __launch_bounds__(128) void tma_cooperative_gemm(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b, float* out, int split) {
  static_assert(Stages == 2 || Stages == 4);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& smem = *reinterpret_cast<HopperStorage<BN, Stages, BatchK>*>(storage);
  const int n = (blockIdx.x % (N / BN)) * BN;
  const int slice = blockIdx.x / (N / BN);
  const int first = slice * (K / split);
  const int steps = (K / split) / (HopperBK * BatchK);
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < Stages; ++s) barrier_init(&smem.full[s], 1);
    barrier_init_fence();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < Stages - 1; ++s)
      if (s < steps)
        load_tma_batch(smem, s, &map_a, &map_b, first + s * HopperBK * BatchK, n);
  }
  SteadyWgmmaAccumulators<BN> acc;
  acc.clear();
  for (int step = 0; step < steps; ++step) {
    const int future = step + Stages - 1;
    if (threadIdx.x == 0 && future < steps)
      load_tma_batch(smem, future & (Stages - 1), &map_a, &map_b,
                     first + future * HopperBK * BatchK, n);
    const int slot = step & (Stages - 1);
    barrier_wait(&smem.full[slot], (step / Stages) & 1);
    consume_tma_batch(acc, smem, slot);
    __syncthreads();  // Current slot can be refilled on the next iteration.
  }
  acc.store(out + size_t(slice) * ElementsC + n);
}

template <int BN, int Stages, int BatchK>
inline int configure_tma_cooperative() {
  constexpr size_t bytes = sizeof(HopperStorage<BN, Stages, BatchK>);
  CUDA_CHECK(cudaFuncSetAttribute(tma_cooperative_gemm<BN, Stages, BatchK>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  CUDA_CHECK(cudaFuncSetAttribute(tma_cooperative_gemm<BN, Stages, BatchK>,
      cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  int resident = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
      tma_cooperative_gemm<BN, Stages, BatchK>, 128, bytes));
  return resident;
}

}  // namespace narrow
