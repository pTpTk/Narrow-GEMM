#pragma once
#include "hopper_ptx.cuh"

// The PTX register lists are explicit because inline assembly cannot accept
// a C++ array as a WGMMA register tuple. No generated library code is included.
namespace narrow {
template <int BN> struct WgmmaInstruction;

template <> struct WgmmaInstruction<32> {
  template <bool Accumulate>
  __device__ __forceinline__ static void run(float (&d)[16], uint64_t a, uint64_t b) {
    if constexpr (Accumulate) {
      asm volatile(
          "{ .reg .pred use_d;\n"
          "setp.ne.u32 use_d, 1, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,"
          "%8,%9,%10,%11,%12,%13,%14,%15},"
          " %16, %17, use_d, 1, 1, 0, 0;\n }"
          : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
            "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
            "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
            "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
          : "l"(a), "l"(b) : "memory");
    }
    else {
      asm volatile(
          "{ .reg .pred use_d;\n"
          "setp.ne.u32 use_d, 0, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,"
          "%8,%9,%10,%11,%12,%13,%14,%15},"
          " %16, %17, use_d, 1, 1, 0, 0;\n }"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
            "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
            "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
            "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15])
          : "l"(a), "l"(b) : "memory");
    }
  }
};

template <> struct WgmmaInstruction<64> {
  template <bool Accumulate>
  __device__ __forceinline__ static void run(float (&d)[32], uint64_t a, uint64_t b) {
    if constexpr (Accumulate) {
      asm volatile(
          "{ .reg .pred use_d;\n"
          "setp.ne.u32 use_d, 1, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,"
          "%8,%9,%10,%11,%12,%13,%14,%15,"
          "%16,%17,%18,%19,%20,%21,%22,%23,"
          "%24,%25,%26,%27,%28,%29,%30,%31},"
          " %32, %33, use_d, 1, 1, 0, 0;\n }"
          : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
            "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
            "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
            "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
            "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
            "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
            "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
            "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
          : "l"(a), "l"(b) : "memory");
    }
    else {
      asm volatile(
          "{ .reg .pred use_d;\n"
          "setp.ne.u32 use_d, 0, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,"
          "%8,%9,%10,%11,%12,%13,%14,%15,"
          "%16,%17,%18,%19,%20,%21,%22,%23,"
          "%24,%25,%26,%27,%28,%29,%30,%31},"
          " %32, %33, use_d, 1, 1, 0, 0;\n }"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
            "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
            "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
            "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
            "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
            "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
            "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
            "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
          : "l"(a), "l"(b) : "memory");
    }
  }
};

template <> struct WgmmaInstruction<128> {
  template <bool Accumulate>
  __device__ __forceinline__ static void run(float (&d)[64], uint64_t a, uint64_t b) {
    if constexpr (Accumulate) {
      asm volatile(
          "{ .reg .pred use_d;\n"
          "setp.ne.u32 use_d, 1, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,"
          "%8,%9,%10,%11,%12,%13,%14,%15,"
          "%16,%17,%18,%19,%20,%21,%22,%23,"
          "%24,%25,%26,%27,%28,%29,%30,%31,"
          "%32,%33,%34,%35,%36,%37,%38,%39,"
          "%40,%41,%42,%43,%44,%45,%46,%47,"
          "%48,%49,%50,%51,%52,%53,%54,%55,"
          "%56,%57,%58,%59,%60,%61,%62,%63},"
          " %64, %65, use_d, 1, 1, 0, 0;\n }"
          : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
            "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
            "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
            "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
            "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
            "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
            "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
            "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
            "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
            "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
            "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
            "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
            "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
            "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
            "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
            "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
          : "l"(a), "l"(b) : "memory");
    }
    else {
      asm volatile(
          "{ .reg .pred use_d;\n"
          "setp.ne.u32 use_d, 0, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,"
          "%8,%9,%10,%11,%12,%13,%14,%15,"
          "%16,%17,%18,%19,%20,%21,%22,%23,"
          "%24,%25,%26,%27,%28,%29,%30,%31,"
          "%32,%33,%34,%35,%36,%37,%38,%39,"
          "%40,%41,%42,%43,%44,%45,%46,%47,"
          "%48,%49,%50,%51,%52,%53,%54,%55,"
          "%56,%57,%58,%59,%60,%61,%62,%63},"
          " %64, %65, use_d, 1, 1, 0, 0;\n }"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
            "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
            "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
            "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
            "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
            "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
            "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
            "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]),
            "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
            "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
            "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]),
            "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]),
            "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]),
            "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
            "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
            "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
          : "l"(a), "l"(b) : "memory");
    }
  }
};

template <int BN> struct WgmmaAccumulators {
  float d[BN / 2];  // First MMA defines these using scale-D=0; no zero-fill.

  __device__ __forceinline__ void compiler_fence() {
    #pragma unroll
    for (int i = 0; i < BN / 2; ++i)
      asm volatile("" : "+f"(d[i]) :: "memory");
  }

  __device__ __forceinline__ void consume(const bf16* a, const bf16* b, bool first) {
    if (!first) compiler_fence();
    wgmma_fence();
    if (first)
      WgmmaInstruction<BN>::template run<false>(d, wgmma_descriptor(a), wgmma_descriptor(b));
    else
      WgmmaInstruction<BN>::template run<true>(d, wgmma_descriptor(a), wgmma_descriptor(b));
    #pragma unroll
    for (int kk = 16; kk < HopperBK; kk += 16)
      WgmmaInstruction<BN>::template run<true>(
          d, wgmma_descriptor(a + kk), wgmma_descriptor(b + kk));
    wgmma_commit();
    wgmma_wait();  // Commit alone does NOT end the shared operand lifetime.
    compiler_fence();
  }

  __device__ __forceinline__ void store(float* output) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row = warp * 16 + lane / 4;
    #pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int col = j * 8 + (lane % 4) * 2;
      output[row * N + col] = d[4 * j];
      output[row * N + col + 1] = d[4 * j + 1];
      output[(row + 8) * N + col] = d[4 * j + 2];
      output[(row + 8) * N + col + 1] = d[4 * j + 3];
    }
  }
};

// Tuned consumers keep one accumulator definition and a uniform sequence of
// accumulating instructions. Initialization happens outside the async group;
// there is no scale-D branch defining a different tuple inside the group.
// Stage 5 retains the original consume() above as the measured reference.
template <int BN> struct SteadyWgmmaAccumulators : WgmmaAccumulators<BN> {
  __device__ __forceinline__ void clear() {
    #pragma unroll
    for (int i = 0; i < BN / 2; ++i) this->d[i] = 0.f;
  }

  __device__ __forceinline__ void begin() {
    this->compiler_fence();
    wgmma_fence();
  }

  __device__ __forceinline__ void issue(const bf16* a, const bf16* b) {
    #pragma unroll
    for (int kk = 0; kk < HopperBK; kk += 16)
      WgmmaInstruction<BN>::template run<true>(
          this->d, wgmma_descriptor(a + kk), wgmma_descriptor(b + kk));
  }

  __device__ __forceinline__ void finish() {
    wgmma_commit();
    wgmma_wait();
    this->compiler_fence();
  }
};

// Batch independent 64-wide TMA boxes under one completion barrier and one
// WGMMA commit/wait. The 128B swizzle box stays valid even when BatchK=2.
template <int BN, int Stages, int BatchK = 1> struct alignas(1024) HopperStorage {
  bf16 a[Stages][BatchK * BM * HopperBK];
  bf16 b[Stages][BatchK * BN * HopperBK];
  uint64_t full[Stages];
  uint64_t empty[Stages];
};

}  // namespace narrow
