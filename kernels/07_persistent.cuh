#pragma once
#include "06_warp_specialized.cuh"
#include <algorithm>

namespace narrow {

// The only plan data is our own pair of hardware descriptors and launch sizes.
// No library scheduler, layout object, pipeline, or GEMM operator is called.
template <int BN>
inline Launch prepare_hopper(int stage, const bf16* a, const bf16* b, float* c,
                             float* partial, int split, cudaStream_t stream) {
  const CUtensorMap map_a = encode_tensor_map(a, M, BM);
  const CUtensorMap map_b = encode_tensor_map(b, N, BN);
  float* target = split == 1 ? c : partial;
  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const int jobs = (N / BN) * split;
  const int workers = stage == 7 ? std::min(jobs, properties.multiProcessorCount) : jobs;
  constexpr int Stages = 3;  // Exercise: try 4, 5, 6, then check shared residency.
  if (stage == 5) configure_tma_serial<BN>();
  else if (stage == 6) configure_tma_pipeline<BN, Stages, false, false>();
  else configure_tma_pipeline<BN, Stages, true, true>();
  std::fprintf(stderr, "Hopper stage=%d BN=%d jobs=%d workers=%d stages=%d shared_bytes=%zu\n",
      stage, BN, jobs, workers, stage == 5 ? 1 : Stages,
      stage == 5 ? sizeof(HopperStorage<BN, 1>) : sizeof(HopperStorage<BN, Stages>));
  return [=] {
    if (stage == 5) launch_tma_serial<BN>(map_a, map_b, target, split, stream);
    else if (stage == 6)
      launch_tma_pipeline<BN, Stages, false, false>(map_a, map_b, target, split, workers, stream);
    else
      launch_tma_pipeline<BN, Stages, true, true>(map_a, map_b, target, split, workers, stream);
    if (split > 1) launch_reduce(partial, c, split, stream);
  };
}

}  // namespace narrow
