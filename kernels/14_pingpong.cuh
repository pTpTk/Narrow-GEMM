#pragma once
#include "09_advanced.cuh"

namespace narrow {

// Two whole-tile consumer WGs + one full producer WG. A math token alternates
// whole output jobs. While one WG runs MMA, the other stages/stores its output.
// All 128 threads of each WG participate in setmaxnreg; a 32-thread producer
// cannot legally use the full-WG redistribution instruction.
template <int BN, int BatchK>
__global__ __launch_bounds__(384, 1) void pingpong_gemm(
    const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b,
    const __grid_constant__ CUtensorMap map_c, int split, int slots) {
  extern __shared__ __align__(1024) unsigned char bytes[];
  AdvancedStorage<BN, BatchK, 2> s{bytes, slots};
  if (threadIdx.x == 0) {
    prefetch_tensor_map(&map_a); prefetch_tensor_map(&map_b); prefetch_tensor_map(&map_c);
    for (int g = 0; g < 2; ++g) for (int i = 0; i < slots; ++i) {
      barrier_init(s.full(g, i), 1); barrier_init(s.empty(g, i), 4);
    }
    *s.control(0) = 0;
    barrier_init_fence();
  }
  __syncthreads();
  const int jobs = (N / BN) * split, steps = K / (HopperBK * BatchK * split);
  if (threadIdx.x >= 256) {
    registers_release<40>();
    if (threadIdx.x == 256) {
      int job[2] = {int(blockIdx.x) * 2, int(blockIdx.x) * 2 + 1};
      int tick[2] = {0, 0}, step[2] = {0, 0};
      // Fair, nonblocking probing: a busy WG's ring cannot prevent prefetch
      // for its peer, including while the former is running its epilogue.
      while (job[0] < jobs || job[1] < jobs) {
        for (int g = 0; g < 2; ++g) if (job[g] < jobs) {
          const int slot = tick[g] % slots;
          if (tick[g] >= slots && !barrier_ready(s.empty(g, slot), ((tick[g] / slots) - 1) & 1)) continue;
          const int first = (job[g] / (N / BN)) * steps;
          load_advanced(s, g, tick[g], first + step[g], job[g] % (N / BN), &map_a, &map_b);
          ++tick[g];
          if (++step[g] == steps) { step[g] = 0; job[g] += int(gridDim.x) * 2; }
        }
      }
    }
  } else {
    registers_acquire<(BN == 256 ? 192 : 128)>();
    const int group = threadIdx.x / 128, tid = threadIdx.x % 128;
    AdvancedAccumulator<BN> acc; int tick = 0;
    for (int job = int(blockIdx.x) * 2 + group; job < jobs; job += int(gridDim.x) * 2) {
      if (tid == 0) {
        volatile unsigned* turn = s.control(0);
        while (*turn != unsigned(group)) {}
      }
      consumer_sync(group);
      compute_advanced<false, true>(acc, s, group, tick, steps);
      consumer_sync(group);
      if (tid == 0) *reinterpret_cast<volatile unsigned*>(s.control(0)) = 1 - group;
      store_advanced_tma(acc, s, group, &map_c, job % (N / BN), job / (N / BN));
    }
    if (tid == 0) tma_store_wait_all();
  }
  __syncthreads();
}

}  // namespace narrow
