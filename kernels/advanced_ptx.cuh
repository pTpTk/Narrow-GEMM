#pragma once
#include "wgmma.cuh"

namespace narrow {

__device__ __forceinline__ void prefetch_tensor_map(const CUtensorMap* p) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(p) : "memory");
}
template <int Pending>
__device__ __forceinline__ void wgmma_wait_pending() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(Pending) : "memory");
}
__device__ __forceinline__ void consumer_sync(int group = 0) {
  // 128 consuming threads; producer threads must not participate here.
  asm volatile("bar.sync %0, 128;" :: "r"(group + 1) : "memory");
}
__device__ __forceinline__ bool barrier_ready(uint64_t* p, unsigned phase) {
  unsigned ready;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
               "selp.b32 %0, 1, 0, p; }"
               : "=r"(ready) : "r"(shared_address(p)), "r"(phase) : "memory");
  return ready;
}
__device__ __forceinline__ unsigned cluster_rank() {
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ void barrier_arrive_remote(uint64_t* p, unsigned rank) {
  asm volatile("{ .reg .b32 remote; mapa.shared::cluster.u32 remote, %0, %1; "
               "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote]; }"
               :: "r"(shared_address(p)), "r"(rank) : "memory");
}
__device__ __forceinline__ void tma_load_3d(
    bf16* dst, const CUtensorMap* map, int row, int k_atom, uint64_t* full) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
               "[%0], [%1, {0, %2, %3}], [%4];"
               :: "r"(shared_address(dst)), "l"(map), "r"(row), "r"(k_atom),
                  "r"(shared_address(full)) : "memory");
}
__device__ __forceinline__ void tma_load_3d_multicast(
    bf16* dst, const CUtensorMap* map, int k_atom, uint64_t* full) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
               "[%0], [%1, {0, 0, %2}], [%3], %4;"
               :: "r"(shared_address(dst)), "l"(map), "r"(k_atom),
                  "r"(shared_address(full)), "h"(uint16_t(3)) : "memory");
}
__device__ __forceinline__ void tma_store_2d(
    const CUtensorMap* map, float* src, int n, int row) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
               "[%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(shared_address(src)), "r"(n), "r"(row) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ unsigned publish_partial(unsigned* counter) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
               : "=r"(old) : "l"(counter) : "memory");
  return old;
}
__device__ __forceinline__ float2 load_partial(const float* p) {
  float2 value;
  // Bypass per-SM L1, including stale cache lines from an earlier graph replay.
  asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];"
               : "=f"(value.x), "=f"(value.y) : "l"(p) : "memory");
  return value;
}

inline CUtensorMap encode_tensor_map_3d(const bf16* data, int rows, int tile_rows, int batch) {
  CUtensorMap result{};
  const cuuint64_t dims[3] = {HopperBK, static_cast<cuuint64_t>(rows), K / HopperBK};
  const cuuint64_t strides[2] = {K * sizeof(bf16), HopperBK * sizeof(bf16)};
  const cuuint32_t box[3] = {HopperBK, static_cast<cuuint32_t>(tile_rows), static_cast<cuuint32_t>(batch)};
  const cuuint32_t elements[3] = {1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
      const_cast<bf16*>(data), dims, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return result;
}
inline CUtensorMap encode_output_map(float* data, int bn, int split) {
  CUtensorMap result{};
  const cuuint64_t dims[2] = {N, static_cast<cuuint64_t>(M * split)};
  const cuuint64_t strides[1] = {N * sizeof(float)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(bn), M};
  const cuuint32_t elements[2] = {1, 1};
  driver_check(cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
      data, dims, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return result;
}

template <int BN> struct AdvancedAccumulator : WgmmaAccumulators<BN> {
  __device__ __forceinline__ uint64_t descriptor(const bf16* p) {
    unsigned lo = (shared_address(p) >> 4) | (1u << 16);
    lo = __shfl_sync(0xffffffff, lo, 0);
    return (1ull << 62) | (64ull << 32) | lo;
  }
  template <bool First>
  __device__ __forceinline__ void issue_atom(const bf16* a, const bf16* b) {
    const uint64_t ad = descriptor(a), bd = descriptor(b);
    WgmmaInstruction<BN>::template run<!First>(this->d, ad, bd);
    #pragma unroll
    for (int kk = 16; kk < HopperBK; kk += 16)
      WgmmaInstruction<BN>::template run<true>(this->d, ad + kk / 8, bd + kk / 8);
  }
  template <bool First, int BatchK>
  __device__ __forceinline__ void issue_group(const bf16* a, const bf16* b) {
    if constexpr (!First) this->compiler_fence();
    wgmma_fence();
    issue_atom<First>(a, b);
    #pragma unroll
    for (int i = 1; i < BatchK; ++i)
      issue_atom<false>(a + i * BM * HopperBK, b + i * BN * HopperBK);
    wgmma_commit();
  }
  template <bool Shared>
  __device__ __forceinline__ void store_pairs(float* output, int stride) {
    const int tid = threadIdx.x % 128, row = (tid / 32) * 16 + (tid % 32) / 4;
    #pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int col = j * 8 + (tid % 4) * 2;
      if constexpr (Shared) {
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(shared_address(output + row * stride + col)),
                     "f"(this->d[4*j]), "f"(this->d[4*j+1]) : "memory");
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(shared_address(output + (row+8) * stride + col)),
                     "f"(this->d[4*j+2]), "f"(this->d[4*j+3]) : "memory");
      } else {
        asm volatile("st.global.v2.f32 [%0], {%1,%2};" :: "l"(output + row * stride + col),
                     "f"(this->d[4*j]), "f"(this->d[4*j+1]) : "memory");
        asm volatile("st.global.v2.f32 [%0], {%1,%2};" :: "l"(output + (row+8) * stride + col),
                     "f"(this->d[4*j+2]), "f"(this->d[4*j+3]) : "memory");
      }
    }
  }
};

}  // namespace narrow
