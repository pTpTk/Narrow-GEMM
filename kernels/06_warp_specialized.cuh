#pragma once
#include "05_tma_wgmma.cuh"

namespace narrow {

// One group is BatchK independent 64-wide TMA boxes. A single transaction
// barrier tracks their combined bytes; the hardware swizzle box stays 128B.
template <int BN, int Stages, int BatchK>
__device__ __forceinline__ void load_tma_batch(
    HopperStorage<BN, Stages, BatchK>& smem, int slot,
    const CUtensorMap* map_a, const CUtensorMap* map_b, int k, int n) {
  barrier_expect(&smem.full[slot], BatchK * (BM + BN) * HopperBK * sizeof(bf16));
  #pragma unroll
  for (int b = 0; b < BatchK; ++b) {
    tma_load_2d(smem.a[slot] + b * BM * HopperBK, map_a,
                k + b * HopperBK, 0, &smem.full[slot]);
    tma_load_2d(smem.b[slot] + b * BN * HopperBK, map_b,
                k + b * HopperBK, n, &smem.full[slot]);
  }
}

template <int BN, int Stages, int BatchK>
__device__ __forceinline__ void consume_tma_batch(
    SteadyWgmmaAccumulators<BN>& acc,
    const HopperStorage<BN, Stages, BatchK>& smem, int slot) {
  acc.begin();
  #pragma unroll
  for (int b = 0; b < BatchK; ++b)
    acc.issue(smem.a[slot] + b * BM * HopperBK, smem.b[slot] + b * BN * HopperBK);
  acc.finish();  // All WGMMA reads complete before this slot may be reused.
}

// 128 consumers + ONE producer warp. A full idle producer warpgroup costs
// registers/threads on a narrow memory-bound GEMM without doing more TMA work.
template <int BN, int Stages, int BatchK, bool Persistent>
__global__ __launch_bounds__(160) void tma_pipeline_gemm(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b, float* out, int split) {
  static_assert(Stages == 2 || Stages == 4);
  static_assert(BatchK == 1 || BatchK == 2);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& smem = *reinterpret_cast<HopperStorage<BN, Stages, BatchK>*>(storage);
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < Stages; ++s) {
      barrier_init(&smem.full[s], 1);
      barrier_init(&smem.empty[s], 4);  // One release per consumer warp.
    }
    barrier_init_fence();
  }
  __syncthreads();
  const int jobs = (N / BN) * split;
  const int steps = (K / split) / (HopperBK * BatchK);
  const int job_stride = Persistent ? int(gridDim.x) : jobs;

  if (threadIdx.x >= 128) {
    if (threadIdx.x == 128) {
      int tick = 0;
      for (int job = blockIdx.x; job < jobs; job += job_stride) {
        const int n = (job % (N / BN)) * BN;
        const int first = (job / (N / BN)) * (K / split);
        for (int step = 0; step < steps; ++step, ++tick) {
          const int slot = tick & (Stages - 1);
          if (tick >= Stages)
            barrier_wait(&smem.empty[slot], ((tick / Stages) - 1) & 1);
          load_tma_batch(smem, slot, &map_a, &map_b,
                         first + step * HopperBK * BatchK, n);
        }
      }
    }
  } else {
    SteadyWgmmaAccumulators<BN> acc;
    int tick = 0;
    for (int job = blockIdx.x; job < jobs; job += job_stride) {
      acc.clear();  // Outside every asynchronous WGMMA group.
      for (int step = 0; step < steps; ++step, ++tick) {
        const int slot = tick & (Stages - 1);
        barrier_wait(&smem.full[slot], (tick / Stages) & 1);
        consume_tma_batch(acc, smem, slot);
        // Each warp releases only after all its lanes finish the MMA wait.
        if ((threadIdx.x & 31) == 0) barrier_arrive(&smem.empty[slot]);
      }
      const int n = (job % (N / BN)) * BN, slice = job / (N / BN);
      acc.store(out + size_t(slice) * ElementsC + n);
    }
  }
  __syncthreads();
}

template <int BN, int Stages, int BatchK, bool Persistent>
inline int configure_tma_pipeline() {
  constexpr size_t bytes = sizeof(HopperStorage<BN, Stages, BatchK>);
  CUDA_CHECK(cudaFuncSetAttribute(tma_pipeline_gemm<BN, Stages, BatchK, Persistent>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  // Inputs arrive through TMA rather than L1. Prefer shared capacity so the
  // occupancy calculation and worker grid can use several buffered CTAs.
  CUDA_CHECK(cudaFuncSetAttribute(tma_pipeline_gemm<BN, Stages, BatchK, Persistent>,
      cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  int resident = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
      tma_pipeline_gemm<BN, Stages, BatchK, Persistent>, 160, bytes));
  return resident;
}

template <int BN, int Stages, int BatchK, bool Persistent>
inline void launch_tma_pipeline(CUtensorMap a, CUtensorMap b, float* out,
                                int split, int workers, cudaStream_t stream) {
  tma_pipeline_gemm<BN, Stages, BatchK, Persistent><<<workers, 160,
      sizeof(HopperStorage<BN, Stages, BatchK>), stream>>>(a, b, out, split);
}

}  // namespace narrow
