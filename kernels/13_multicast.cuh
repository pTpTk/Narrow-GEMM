#pragma once
#include "09_advanced.cuh"

namespace narrow {

// The two CTAs compute adjacent N tiles. They share A, never B. Full barriers
// have identical shared addresses in both CTAs. The extra 'armed' handshake
// guarantees that both expect_tx operations precede the A multicast.
template <int BN, int BatchK>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(160) void multicast_gemm(
    const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b,
    float* out, int split, int slots) {
  extern __shared__ __align__(1024) unsigned char bytes[];
  AdvancedStorage<BN, BatchK> s{bytes, slots};
  const unsigned rank = cluster_rank();
  if (threadIdx.x == 0) {
    prefetch_tensor_map(&map_a); prefetch_tensor_map(&map_b);
    for (int i = 0; i < slots; ++i) {
      barrier_init(s.full(0, i), 1);
      barrier_init(s.empty(0, i), 8); // Four warps from each CTA.
      barrier_init(s.armed(0, i), 2); // Two producer leaders on rank zero.
    }
    barrier_init_fence();
  }
  __syncthreads();
  cluster_sync();
  WorkIterator<BN, BatchK> iter(blockIdx.x, gridDim.x, split, false);
  WorkItem w; int tick = 0;
  if (threadIdx.x == 128) {
    while (iter.next(w)) for (int k = 0; k < w.count; ++k, ++tick) {
      const int slot = tick % slots;
      if (tick >= slots) barrier_wait(s.empty(0, slot), ((tick / slots) - 1) & 1);
      barrier_expect(s.full(0, slot), s.ABytes + s.BBytes);
      tma_load_3d(s.b(0, slot), &map_b, w.tile * BN, (w.first + k) * BatchK, s.full(0, slot));
      barrier_arrive_remote(s.armed(0, slot), 0);
      if (rank == 0) {
        barrier_wait(s.armed(0, slot), (tick / slots) & 1);
        tma_load_3d_multicast(s.a(0, slot), &map_a, (w.first + k) * BatchK, s.full(0, slot));
      }
    }
    // Wait for every final shared-memory reader, including those in the peer
    // CTA. No CTA can retire while remote TMA/barrier operations still use it.
    for (int slot = 0; slot < slots && slot < tick; ++slot) {
      const int last = slot + ((tick - 1 - slot) / slots) * slots;
      barrier_wait(s.empty(0, slot), (last / slots) & 1);
    }
  } else if (threadIdx.x < 128) {
    AdvancedAccumulator<BN> acc;
    while (iter.next(w)) {
      compute_advanced<true, true>(acc, s, 0, tick, w.count);
      acc.template store_pairs<false>(out + size_t(w.slice) * ElementsC + w.tile * BN, N);
    }
  }
  __syncthreads();
  cluster_sync();
}

}  // namespace narrow
