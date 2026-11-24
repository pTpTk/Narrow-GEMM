#pragma once
#include "05_tma_wgmma.cuh"

namespace narrow {

// Stage 6: one logical job per CTA, no register redistribution.
// Stage 7: bounded persistent workers + asymmetric register budgets.
// Redistribute is separate so editing stage 7's alias can isolate the changes.
template <int BN, int Stages, bool Persistent, bool Redistribute>
__global__ __launch_bounds__(256, 1) void tma_pipeline_gemm(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b, float* out, int split) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& smem = *reinterpret_cast<HopperStorage<BN, Stages>*>(storage);
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < Stages; ++s) {
      barrier_init(&smem.full[s], 1);     // One TMA issuer arrives + expects bytes.
      barrier_init(&smem.empty[s], 128);  // All consumer threads release a slot.
    }
    barrier_init_fence();
  }
  __syncthreads();
  const int jobs = (N / BN) * split;
  const int steps = (K / split) / HopperBK;
  // Persistent workers jump by the launched worker count. In stage 6 the
  // grid already contains every job, so the same loop executes once per CTA.
  const int job_stride = Persistent ? int(gridDim.x) : jobs;

  if (threadIdx.x >= 128) {
    if constexpr (Redistribute) registers_release<40>();
    // Elect a fixed lane of the producer warpgroup; other producer lanes
    // remain available for register release and the final block barrier.
    if (threadIdx.x == 128) {
      int tick = 0;
      for (int job = blockIdx.x; job < jobs; job += job_stride) {
        const int n_tile = job % (N / BN), slice = job / (N / BN);
        for (int k_step = 0; k_step < steps; ++k_step, ++tick) {
          const int slot = tick % Stages;
          // Initial slots are free. Thereafter wait for the previous use's
          // consumer arrivals, even when it belonged to a different job.
          if (tick >= Stages)
            barrier_wait(&smem.empty[slot], ((tick / Stages) - 1) & 1);
          barrier_expect(&smem.full[slot], (BM + BN) * HopperBK * sizeof(bf16));
          const int k = slice * (K / split) + k_step * HopperBK;
          tma_load_2d(smem.a[slot], &map_a, k, 0, &smem.full[slot]);
          tma_load_2d(smem.b[slot], &map_b, k, n_tile * BN, &smem.full[slot]);
        }
      }
    }
  } else {
    if constexpr (Redistribute) registers_acquire<128>();
    WgmmaAccumulators<BN> acc;
    int tick = 0;
    for (int job = blockIdx.x; job < jobs; job += job_stride) {
      for (int k_step = 0; k_step < steps; ++k_step, ++tick) {
        const int slot = tick % Stages;
        barrier_wait(&smem.full[slot], (tick / Stages) & 1);
        acc.consume(smem.a[slot], smem.b[slot], k_step == 0);
        // consume() includes WGMMA wait<0>; every consumer then arrives.
        // Producer can overwrite only after all 128 arrivals complete.
        barrier_arrive(&smem.empty[slot]);
      }
      const int n_tile = job % (N / BN), slice = job / (N / BN);
      acc.store(out + size_t(slice) * ElementsC + n_tile * BN);
    }
  }
  // All issued loads have a consumer and all WGMMA groups are drained.
  // No CTA exits while its producer still has an outstanding transfer.
  __syncthreads();
}

template <int BN, int Stages, bool Persistent, bool Redistribute>
inline void configure_tma_pipeline() {
  CUDA_CHECK(cudaFuncSetAttribute(tma_pipeline_gemm<BN, Stages, Persistent, Redistribute>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(HopperStorage<BN, Stages>)));
}

template <int BN, int Stages, bool Persistent, bool Redistribute>
inline void launch_tma_pipeline(CUtensorMap a, CUtensorMap b, float* out,
                                int split, int workers, cudaStream_t stream) {
  tma_pipeline_gemm<BN, Stages, Persistent, Redistribute><<<workers, 256,
      sizeof(HopperStorage<BN, Stages>), stream>>>(a, b, out, split);
}

}  // namespace narrow
