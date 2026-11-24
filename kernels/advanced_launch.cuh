#pragma once
#include "13_multicast.cuh"
#include "14_pingpong.cuh"

namespace narrow {

template <class Kernel>
inline int configure_advanced(Kernel kernel, int threads, size_t bytes,
                              const cudaDeviceProp& prop) {
  if (bytes > prop.sharedMemPerBlockOptin)
    throw UnsupportedConfiguration("input/output rings exceed this device's per-CTA shared-memory limit");
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  int resident = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, threads, bytes));
  if (!resident) throw UnsupportedConfiguration("configuration has no resident CTA");
  return resident;
}

template <int BN, int BatchK>
inline Launch prepare_advanced_variant(int mode, const bf16* a, const bf16* b, float* c,
    float* partial, int split, int slots, int worker_cap, cudaStream_t stream) {
  int device = 0; cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const CUtensorMap ma = encode_tensor_map_3d(a, M, BM, BatchK);
  const CUtensorMap mb = encode_tensor_map_3d(b, N, BN, BatchK);
  // Used only by TMA epilogues; encoding does not touch the output data.
  const CUtensorMap mc = encode_output_map(split == 1 ? c : partial, BN, split);
  const int jobs = N / BN * split;
  int resident = 0, workers = 0;
  size_t bytes = AdvancedStorage<BN, BatchK>::bytes(slots, mode == 11);
  if (mode == 14) {
    bytes = AdvancedStorage<BN, BatchK, 2>::bytes(slots, true);
    // Reserve enough SM shared capacity to admit one CTA only. This also
    // guarantees a register pool for the 2x consumer setmaxnreg.inc requests.
    bytes = std::max(bytes, size_t(prop.sharedMemPerMultiprocessor / 2 + 1024));
    resident = configure_advanced(pingpong_gemm<BN, BatchK>, 384, bytes, prop);
    workers = std::min((jobs + 1) / 2, prop.multiProcessorCount);
  } else if (mode == 13) {
    int supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device));
    if (!supported) throw UnsupportedConfiguration("two-CTA clusters are unavailable on this device/MIG configuration");
    resident = configure_advanced(multicast_gemm<BN, BatchK>, 160, bytes, prop);
    const int cap = worker_cap ? std::min(worker_cap, resident) : resident;
    workers = std::min(jobs, prop.multiProcessorCount * cap) & ~1;
    if (workers < 2) throw UnsupportedConfiguration("device cannot provision a two-CTA worker grid");
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(workers); config.blockDim = dim3(160); config.dynamicSmemBytes = bytes;
    int clusters = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, multicast_gemm<BN, BatchK>, &config));
    if (!clusters) throw UnsupportedConfiguration("two-CTA clusters have no legal residency on this device/MIG configuration");
    workers = std::min(workers, clusters * 2);
  } else {
    resident = mode == 9 ? configure_advanced(advanced_gemm<BN, BatchK, false>, 160, bytes, prop)
                         : configure_advanced(advanced_gemm<BN, BatchK, true>, 160, bytes, prop);
    const int cap = worker_cap ? std::min(worker_cap, resident) : resident;
    workers = std::min(jobs, prop.multiProcessorCount * cap);
  }
  std::fprintf(stderr, "Advanced stage=%d BN=%d split_or_worker_budget=%d workers=%d resident_ctas_per_sm=%d tile_k=%d slots=%d shared_bytes=%zu decomposition=%s\n",
      mode, BN, split, workers, resident, HopperBK * BatchK, slots, bytes, mode == 12 ? "stream-k" : "uniform");
  std::shared_ptr<DeviceBuffer<float>> sk_partial;
  std::shared_ptr<DeviceBuffer<unsigned>> counters;
  if (mode == 12) {
    sk_partial = std::make_shared<DeviceBuffer<float>>(size_t(2 * workers) * BM * BN);
    counters = std::make_shared<DeviceBuffer<unsigned>>(N / BN);
    CUDA_CHECK(cudaMemsetAsync(counters->data, 0, (N / BN) * sizeof(unsigned), stream));
    CUDA_CHECK(cudaMemsetAsync(sk_partial->data, 0xff, size_t(2 * workers) * BM * BN * sizeof(float), stream));
  }
  return [=] {
    if (mode == 14) {
      pingpong_gemm<BN, BatchK><<<workers, 384, bytes, stream>>>(ma, mb, mc, split, slots);
    } else if (mode == 13) {
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(workers); config.blockDim = dim3(160);
      config.dynamicSmemBytes = bytes; config.stream = stream;
      CUDA_CHECK(cudaLaunchKernelEx(&config, multicast_gemm<BN, BatchK>, ma, mb,
                                   split == 1 ? c : partial, split, slots));
    } else if (mode == 9) {
      advanced_gemm<BN, BatchK, false><<<workers, 160, bytes, stream>>>(ma, mb, mc, c, partial, nullptr, split, slots, mode);
    } else {
      advanced_gemm<BN, BatchK, true><<<workers, 160, bytes, stream>>>(ma, mb, mc, c,
          mode == 12 ? sk_partial->data : partial, mode == 12 ? counters->data : nullptr,
          split, slots, mode);
    }
    if (mode != 12 && split > 1) launch_reduce(partial, c, split, stream);
  };
}

template <int BN>
inline Launch prepare_advanced(int mode, const bf16* a, const bf16* b, float* c,
    float* partial, int split, int slots, int batch, int cap, cudaStream_t stream) {
  if (batch == 1) return prepare_advanced_variant<BN, 1>(mode, a, b, c, partial, split, slots, cap, stream);
  return prepare_advanced_variant<BN, 2>(mode, a, b, c, partial, split, slots, cap, stream);
}

}  // namespace narrow
