#pragma once
#include "08_cooperative.cuh"
#include <algorithm>

namespace narrow {

template <int BN, int Stages, int BatchK>
inline Launch prepare_hopper_variant(int stage, CUtensorMap map_a, CUtensorMap map_b,
                                     float* c, float* partial, int split,
                                     int workers_per_sm, cudaStream_t stream) {
  float* target = split == 1 ? c : partial;
  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const int jobs = (N / BN) * split;
  int resident = 0;
  if (stage == 6) resident = configure_tma_pipeline<BN, Stages, BatchK, false>();
  else if (stage == 7) resident = configure_tma_pipeline<BN, Stages, BatchK, true>();
  else resident = configure_tma_cooperative<BN, Stages, BatchK>();
  // Query the MIG device, not the full chip's SM count or an assumed SKU.
  // A memory-bound worker grid needs several resident CTAs per SM when legal.
  const int cap = workers_per_sm ? std::min(workers_per_sm, resident) : resident;
  if (cap < 1) {
    std::fprintf(stderr, "No resident CTA for this shared-memory configuration\n");
    std::exit(EXIT_FAILURE);
  }
  const int workers = stage == 7 ? std::min(jobs, properties.multiProcessorCount * cap) : jobs;
  std::fprintf(stderr,
      "Hopper stage=%d BN=%d jobs=%d workers=%d resident_ctas_per_sm=%d tile_k=%d stages=%d shared_bytes=%zu\n",
      stage, BN, jobs, workers, resident, HopperBK * BatchK, Stages,
      sizeof(HopperStorage<BN, Stages, BatchK>));
  return [=] {
    if (stage == 6)
      launch_tma_pipeline<BN, Stages, BatchK, false>(map_a, map_b, target, split, workers, stream);
    else if (stage == 7)
      launch_tma_pipeline<BN, Stages, BatchK, true>(map_a, map_b, target, split, workers, stream);
    else
      tma_cooperative_gemm<BN, Stages, BatchK><<<jobs, 128,
          sizeof(HopperStorage<BN, Stages, BatchK>), stream>>>(map_a, map_b, target, split);
    if (split > 1) launch_reduce(partial, c, split, stream);
  };
}

// Stage 5 stays the serial measured reference. Stages 6–8 expose a small,
// explicit candidate set; no selection is presented as a measured optimum.
template <int BN>
inline Launch prepare_hopper(int stage, const bf16* a, const bf16* b, float* c,
                             float* partial, int split, int stages, int batch_k,
                             int workers_per_sm, cudaStream_t stream) {
  const CUtensorMap map_a = encode_tensor_map(a, M, BM);
  const CUtensorMap map_b = encode_tensor_map(b, N, BN);
  if (stage == 5) {
    configure_tma_serial<BN>();
    return [=] {
      launch_tma_serial<BN>(map_a, map_b, split == 1 ? c : partial, split, stream);
      if (split > 1) launch_reduce(partial, c, split, stream);
    };
  }
  if (stages != 2 && stages != 4)
    throw UnsupportedConfiguration("stages 6-8 support two/four slots; use stages 9-14 for deeper queues");
  if (stages == 2 && batch_k == 1)
    return prepare_hopper_variant<BN, 2, 1>(stage, map_a, map_b, c, partial, split, workers_per_sm, stream);
  if (stages == 2 && batch_k == 2)
    return prepare_hopper_variant<BN, 2, 2>(stage, map_a, map_b, c, partial, split, workers_per_sm, stream);
  if (stages == 4 && batch_k == 1)
    return prepare_hopper_variant<BN, 4, 1>(stage, map_a, map_b, c, partial, split, workers_per_sm, stream);
  return prepare_hopper_variant<BN, 4, 2>(stage, map_a, map_b, c, partial, split, workers_per_sm, stream);
}

}  // namespace narrow
