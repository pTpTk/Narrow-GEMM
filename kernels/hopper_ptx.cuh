#pragma once
#include "common.cuh"
#include <cuda.h>
#include <cstdint>

// Direct instruction wrappers only: no CUTLASS, CuTe, DeepGEMM, WMMA, or
// CUDA C++ barrier/pipeline implementations are included by our kernels.
namespace narrow {

constexpr int HopperBK = 64;

__device__ __forceinline__ unsigned shared_address(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void barrier_init(uint64_t* p, int arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(shared_address(p)), "r"(arrivals) : "memory");
}
__device__ __forceinline__ void barrier_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void barrier_expect(uint64_t* p, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(shared_address(p)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void barrier_arrive(uint64_t* p) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :: "r"(shared_address(p)) : "memory");
}
__device__ __forceinline__ void barrier_wait(uint64_t* p, unsigned phase) {
  // A parity wait succeeds once the specified phase has completed, not when
  // it has merely been entered. mbarrier waits have acquire semantics.
  asm volatile(
      "{ .reg .pred done;\n"
      "wait_again:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra wait_again;\n }"
      :: "r"(shared_address(p)), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_load_2d(
    bf16* dst, const CUtensorMap* map, int k, int row, uint64_t* full) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :: "r"(shared_address(dst)), "l"(map), "r"(k), "r"(row),
         "r"(shared_address(full)) : "memory");
}

// K-major BF16, 128-byte swizzle. Each logical row has 64 BF16 values;
// the next group of eight rows begins 1024 bytes later. All stage bases
// are 1024B aligned, making the descriptor's swizzle base-offset field zero.
__device__ __forceinline__ uint64_t wgmma_descriptor(const bf16* p) {
  uint64_t address = shared_address(p);
  return ((address & 0x3ffffull) >> 4) | (1ull << 16) |
         (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

template <int Registers>
__device__ __forceinline__ void registers_release() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(Registers));
}
template <int Registers>
__device__ __forceinline__ void registers_acquire() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(Registers));
}

inline void driver_check(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "CUDA driver: %s (%d)\n", message ? message : "unknown", int(status));
    std::exit(EXIT_FAILURE);
  }
}

// This CUDA driver API encodes a hardware tensor descriptor, not a GEMM.
// Tensor dims are [contiguous K, rows]; both physical row strides remain K.
inline CUtensorMap encode_tensor_map(const bf16* data, int rows, int tile_rows) {
  CUtensorMap result{};
  const cuuint64_t dimensions[2] = {K, static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {K * sizeof(bf16)};
  const cuuint32_t box[2] = {HopperBK, static_cast<cuuint32_t>(tile_rows)};
  const cuuint32_t element_strides[2] = {1, 1};
  driver_check(cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2, const_cast<bf16*>(data), dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return result;
}

}  // namespace narrow
