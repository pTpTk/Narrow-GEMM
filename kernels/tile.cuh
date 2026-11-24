#pragma once
#include "common.cuh"

namespace narrow {

// Padding is deliberately simple; this is NOT a Hopper TMA swizzle.
constexpr int LD = BK + 8;

template <int BN, int Stages> struct alignas(32) Tiles {
  bf16 a[Stages][BM * LD];
  bf16 b[Stages][BN * LD];
};

__device__ __forceinline__ void copy_async_16(void* dst, const void* src) {
  const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
               :: "r"(smem), "l"(src) : "memory");
}
__device__ __forceinline__ void copy_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void copy_wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template <int Rows, bool Async>
__device__ __forceinline__ void copy_rows(bf16* dst, const bf16* src, int k0) {
  // Each lane copies eight adjacent BF16 values. All addresses are 16B aligned.
  // src rows have physical stride K; shared rows have physical stride LD.
  for (int chunk = threadIdx.x; chunk < Rows * BK / 8; chunk += Threads) {
    const int row = chunk / (BK / 8), col = (chunk % (BK / 8)) * 8;
    bf16* to = dst + row * LD + col;
    const bf16* from = src + row * K + k0 + col;
    if constexpr (Async) copy_async_16(to, from);
    else *reinterpret_cast<uint4*>(to) = *reinterpret_cast<const uint4*>(from);
  }
}

template <int BN> struct Accumulators {
  static_assert(BN == 32 || BN == 64 || BN == 128);
  // mma.m16n8k16 distributes 16*8 floats across 32 lanes: four per lane.
  float c[BN / 8][4];

  __device__ void clear() {
    #pragma unroll
    for (int j = 0; j < BN / 8; ++j)
      #pragma unroll
      for (int r = 0; r < 4; ++r) c[j][r] = 0.f;
  }

  __device__ void mma(const bf16* as, const bf16* bs) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      unsigned af[4];
      unsigned ap = static_cast<unsigned>(__cvta_generic_to_shared(
          as + (warp * 16 + lane % 16) * LD + kk + (lane / 16) * 8));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                   : "r"(ap) : "memory");
      #pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        unsigned bf[2];
        unsigned bp = static_cast<unsigned>(__cvta_generic_to_shared(
            bs + (j * 8 + lane % 8) * LD + kk + ((lane / 8) & 1) * 8));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(bf[0]), "=r"(bf[1]) : "r"(bp) : "memory");
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
              "r"(bf[0]), "r"(bf[1]));
      }
    }
  }

  __device__ void store(float* dst) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = warp * 16 + lane / 4;
    #pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int col = j * 8 + (lane % 4) * 2;
      dst[row * N + col] = c[j][0];
      dst[row * N + col + 1] = c[j][1];
      dst[(row + 8) * N + col] = c[j][2];
      dst[(row + 8) * N + col + 1] = c[j][3];
    }
  }
};

}  // namespace narrow
