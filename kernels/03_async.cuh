#pragma once
#include "tile.cuh"

namespace narrow {

// Split=1 is checkpoint 03. Split>1 plus reduction is checkpoint 04.
// Runtime split permits a small sweep without compiling many identical kernels.
template <int BN>
__global__ void async_gemm(const bf16* a, const bf16* b, float* out, int split) {
  extern __shared__ __align__(32) unsigned char warp_storage[];
  auto& smem = *reinterpret_cast<Tiles<BN, 2>*>(warp_storage);
  const int length = K / split;
  const int first = blockIdx.y * length;
  const int iterations = length / BK;
  b += blockIdx.x * BN * K;
  Accumulators<BN> acc;
  acc.clear();

  copy_rows<BM, true>(smem.a[0], a, first);
  copy_rows<BN, true>(smem.b[0], b, first);
  copy_commit();
  copy_wait();
  __syncthreads();

  for (int t = 0; t < iterations; ++t) {
    const int current = t & 1, next = current ^ 1;
    if (t + 1 < iterations) {
      const int k0 = first + (t + 1) * BK;
      copy_rows<BM, true>(smem.a[next], a, k0);
      copy_rows<BN, true>(smem.b[next], b, k0);
      copy_commit();
    }
    // Copies into the OTHER buffer may overlap these synchronous warp MMAs.
    acc.mma(smem.a[current], smem.b[current]);
    if (t + 1 < iterations) copy_wait();
    // wait_group is per issuing thread. The CTA barrier also publishes other
    // threads' copies and protects the buffer that will be reused next time.
    __syncthreads();
  }
  acc.store(out + size_t(blockIdx.y) * ElementsC + blockIdx.x * BN);
}

template <int BN>
inline void configure_async() {
  CUDA_CHECK(cudaFuncSetAttribute(async_gemm<BN>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Tiles<BN, 2>)));
}

template <int BN>
inline void launch_async(const bf16* a, const bf16* b, float* out, int split,
                         cudaStream_t stream) {
  async_gemm<BN><<<dim3(N / BN, split), Threads, sizeof(Tiles<BN, 2>), stream>>>(a, b, out, split);
}

}  // namespace narrow
