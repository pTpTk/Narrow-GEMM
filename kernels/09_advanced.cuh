#pragma once
#include "advanced_ptx.cuh"
#include "04_split_k.cuh"
#include <algorithm>
#include <memory>

namespace narrow {

// Runtime queue depth avoids instantiating a different kernel for every slot
// count. All input atom bases remain 1024B aligned, including BN=16/256.
template <int BN, int BatchK, int Groups = 1> struct AdvancedStorage {
  unsigned char* base;
  int slots;
  static constexpr size_t ABytes = BatchK * BM * HopperBK * sizeof(bf16);
  static constexpr size_t BBytes = BatchK * BN * HopperBK * sizeof(bf16);
  __host__ __device__ static size_t barrier_offset(int s) {
    return Groups * s * (ABytes + BBytes);
  }
  __host__ __device__ static size_t output_offset(int s) {
    return (barrier_offset(s) + Groups * (3 * s + 1) * sizeof(uint64_t) + 1023) & ~size_t(1023);
  }
  __host__ __device__ static size_t bytes(int s, bool epilogue) {
    return output_offset(s) + (epilogue ? Groups * BM * BN * sizeof(float) : 0);
  }
  __device__ bf16* a(int g, int s) {
    return reinterpret_cast<bf16*>(base + (g * slots + s) * ABytes);
  }
  __device__ bf16* b(int g, int s) {
    return reinterpret_cast<bf16*>(base + Groups * slots * ABytes + (g * slots + s) * BBytes);
  }
  __device__ uint64_t* barrier(int g, int kind, int s) {
    return reinterpret_cast<uint64_t*>(base + barrier_offset(slots)) + g * (3 * slots + 1) + kind * slots + s;
  }
  __device__ uint64_t* full(int g, int s) { return barrier(g, 0, s); }
  __device__ uint64_t* empty(int g, int s) { return barrier(g, 1, s); }
  __device__ uint64_t* armed(int g, int s) { return barrier(g, 2, s); }
  __device__ unsigned* control(int g) { return reinterpret_cast<unsigned*>(barrier(g, 3, 0)); }
  __device__ float* output(int g) {
    return reinterpret_cast<float*>(base + output_offset(slots)) + g * BM * BN;
  }
};

struct WorkItem { int tile, first, count, slice; };

// Flatten tile x K-group work for Stream-K. Uniform split-K instead enumerates
// (slice,tile). Producer and consumers use independent copies of this iterator.
template <int BN, int BatchK> struct WorkIterator {
  static constexpr int Tiles = N / BN, KG = K / (HopperBK * BatchK), Units = Tiles * KG;
  int job, stride, split, cursor, end;
  bool stream_k;
  __device__ WorkIterator(int worker, int workers, int slices, bool sk)
      : job(worker), stride(workers), split(slices),
        cursor(Units * worker / workers), end(Units * (worker + 1) / workers), stream_k(sk) {}
  __device__ bool next(WorkItem& w) {
    if (stream_k) {
      if (cursor == end) return false;
      w.tile = cursor / KG; w.first = cursor % KG; w.slice = 0;
      w.count = min(end - cursor, KG - w.first);
      cursor += w.count;
    } else {
      if (job >= Tiles * split) return false;
      w.tile = job % Tiles; w.slice = job / Tiles;
      w.count = KG / split; w.first = w.slice * w.count;
      job += stride;
    }
    return true;
  }
};

template <int BN, int BatchK, int Groups>
__device__ void load_advanced(AdvancedStorage<BN, BatchK, Groups>& s, int group,
                              int tick, int first, int tile,
                              const CUtensorMap* a, const CUtensorMap* b) {
  const int slot = tick % s.slots;
  barrier_expect(s.full(group, slot), s.ABytes + s.BBytes);
  tma_load_3d(s.a(group, slot), a, 0, first * BatchK, s.full(group, slot));
  tma_load_3d(s.b(group, slot), b, tile * BN, first * BatchK, s.full(group, slot));
}

template <bool Cluster, int BN, int BatchK, int Groups>
__device__ void release_advanced(AdvancedStorage<BN, BatchK, Groups>& s, int group, int tick) {
  if ((threadIdx.x & 31) == 0) {
    if constexpr (Cluster) {
      barrier_arrive_remote(s.empty(group, tick % s.slots), 0);
      barrier_arrive_remote(s.empty(group, tick % s.slots), 1);
    } else barrier_arrive(s.empty(group, tick % s.slots));
  }
}

template <bool Cluster, bool Overlap, int BN, int BatchK, int Groups>
__device__ void compute_advanced(AdvancedAccumulator<BN>& acc,
    AdvancedStorage<BN, BatchK, Groups>& s, int group, int& tick, int count) {
  // Explicit prologue: no uninitialized accumulator read, even for a one-group
  // Stream-K segment. Read and release cursors are intentionally different.
  const int begin = tick;
  barrier_wait(s.full(group, tick % s.slots), (tick / s.slots) & 1);
  acc.template issue_group<true, BatchK>(s.a(group, tick % s.slots), s.b(group, tick % s.slots));
  if constexpr (!Overlap) {
    wgmma_wait_pending<0>(); acc.compiler_fence();
    release_advanced<Cluster>(s, group, tick);
  }
  ++tick;
  for (int step = 1; step < count; ++step, ++tick) {
    barrier_wait(s.full(group, tick % s.slots), (tick / s.slots) & 1);
    acc.template issue_group<false, BatchK>(s.a(group, tick % s.slots), s.b(group, tick % s.slots));
    if constexpr (Overlap) {
      wgmma_wait_pending<1>(); acc.compiler_fence();
      release_advanced<Cluster>(s, group, tick - 1);
    } else {
      wgmma_wait_pending<0>(); acc.compiler_fence();
      release_advanced<Cluster>(s, group, tick);
    }
  }
  if constexpr (Overlap) {
    wgmma_wait_pending<0>(); acc.compiler_fence();
    release_advanced<Cluster>(s, group, begin + count - 1);
  }
}

template <int BN, int BatchK, int Groups>
__device__ void store_advanced_tma(AdvancedAccumulator<BN>& acc,
    AdvancedStorage<BN, BatchK, Groups>& s, int group, const CUtensorMap* out,
    int tile, int slice) {
  const int tid = threadIdx.x % 128;
  if (tid == 0) tma_store_wait_read();
  consumer_sync(group);  // Prior async reader must finish before any writer.
  acc.template store_pairs<true>(s.output(group), BN);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  consumer_sync(group);
  if (tid == 0) tma_store_2d(out, s.output(group), tile * BN, slice * M);
  // Persistent next-tile input/MMA work can overlap this store. Only buffer
  // reuse waits for shared reads; kernel exit drains global completion too.
}

template <int BN, int BatchK>
__device__ void stream_k_store(AdvancedAccumulator<BN>& acc,
    AdvancedStorage<BN, BatchK>& s, const WorkItem& w, float* c, float* partial,
    unsigned* counters) {
  constexpr int KG = WorkIterator<BN, BatchK>::KG, U = WorkIterator<BN, BatchK>::Units;
  if (w.first == 0 && w.count == KG) {
    acc.template store_pairs<false>(c + w.tile * BN, N);
    return;
  }
  const int workers = gridDim.x, worker = blockIdx.x;
  const int first_tile = (U * worker / workers) / KG;
  const int side = w.tile == first_tile ? 0 : 1;
  acc.template store_pairs<false>(partial + size_t(2 * worker + side) * BM * BN, BN);
  __threadfence();  // Every writer publishes its own stores, not just lane 0.
  consumer_sync();
  const int first_worker = ((w.tile * KG + 1) * workers - 1) / U;
  const int last_worker = (((w.tile + 1) * KG) * workers - 1) / U;
  if (threadIdx.x == 0)
    *s.control(0) = publish_partial(counters + w.tile) == unsigned(last_worker - first_worker);
  consumer_sync();
  if (*s.control(0)) {
    // Acquire of the last RMW is relayed to all lanes by the named barrier.
    // Fixed worker order makes the reduction independent of completion order.
    for (int pair = threadIdx.x; pair < BM * BN / 2; pair += 128) {
      float2 sum = {0.f, 0.f};
      for (int owner = first_worker; owner <= last_worker; ++owner) {
        const int owner_first = (U * owner / workers) / KG;
        const int slot = 2 * owner + (w.tile == owner_first ? 0 : 1);
        const float2 v = load_partial(partial + size_t(slot) * BM * BN + pair * 2);
        sum.x += v.x; sum.y += v.y;
      }
      *reinterpret_cast<float2*>(c + (pair * 2 / BN) * N + w.tile * BN + pair * 2 % BN) = sum;
    }
    consumer_sync();
    if (threadIdx.x == 0) counters[w.tile] = 0;
  }
  consumer_sync(); // Protect decision word before this WG's next segment.
}

// Stages 9..12 share the same ring and launch resources. The mode changes the
// outstanding-MMA policy, output engine or decomposition, not input precision.
template <int BN, int BatchK, bool Overlap>
__global__ __launch_bounds__(160) void advanced_gemm(
    const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b,
    const __grid_constant__ CUtensorMap map_c, float* c, float* partial, unsigned* counters,
    int split, int slots, int mode) {
  extern __shared__ __align__(1024) unsigned char bytes[];
  AdvancedStorage<BN, BatchK> s{bytes, slots};
  if (threadIdx.x == 0) {
    prefetch_tensor_map(&map_a); prefetch_tensor_map(&map_b);
    if (mode == 11) prefetch_tensor_map(&map_c);
    for (int i = 0; i < slots; ++i) {
      barrier_init(s.full(0, i), 1); barrier_init(s.empty(0, i), 4);
    }
    barrier_init_fence();
  }
  __syncthreads();
  WorkIterator<BN, BatchK> iter(blockIdx.x, gridDim.x, split, mode == 12);
  WorkItem w; int tick = 0;
  if (threadIdx.x == 128) {
    while (iter.next(w)) for (int k = 0; k < w.count; ++k, ++tick) {
      const int slot = tick % slots;
      if (tick >= slots) barrier_wait(s.empty(0, slot), ((tick / slots) - 1) & 1);
      load_advanced(s, 0, tick, w.first + k, w.tile, &map_a, &map_b);
    }
  } else if (threadIdx.x < 128) {
    AdvancedAccumulator<BN> acc;
    while (iter.next(w)) {
      compute_advanced<false, Overlap>(acc, s, 0, tick, w.count);
      if (mode == 12) stream_k_store(acc, s, w, c, partial, counters);
      else if (mode == 11) store_advanced_tma(acc, s, 0, &map_c, w.tile, w.slice);
      else acc.template store_pairs<false>((split == 1 ? c : partial) + size_t(w.slice) * ElementsC + w.tile * BN, N);
    }
    if (mode == 11 && threadIdx.x == 0) tma_store_wait_all();
  }
  __syncthreads();
}

}  // namespace narrow
