#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace narrow {

// The entire exercise specializes one problem. B is physically [N][K].
constexpr int M = 64, N = 4096, K = 8192;
#ifndef NARROW_WARP_BK
#define NARROW_WARP_BK 64
#endif
constexpr int BM = 64, BK = NARROW_WARP_BK, Threads = 128;
static_assert(BK == 32 || BK == 64 || BK == 128);
constexpr size_t ElementsC = size_t(M) * N;
using bf16 = __nv_bfloat16;
using Launch = std::function<void()>;

inline void cuda_check(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA: %s\n", file, line,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(call) ::narrow::cuda_check((call), __FILE__, __LINE__)

template <class T> struct DeviceBuffer {
  T* data = nullptr;
  explicit DeviceBuffer(size_t count) {
    if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
  }
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

}  // namespace narrow
