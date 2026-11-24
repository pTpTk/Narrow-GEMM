#include "common.cuh"
#include "reference.cuh"
#include "kernels/01_scalar.cuh"
#include "kernels/02_mma.cuh"
#include "kernels/03_async.cuh"
#include "kernels/04_split_k.cuh"
#include "kernels/07_persistent.cuh"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace narrow;

struct Options {
  std::string kernel = "all", timing = "graph", pattern = "random", format = "table";
  int bn = 64, split = 4, iterations = 20, samples = 5, warmup = 3;
};

void usage() {
  std::puts("Fixed BF16 GEMM: (M,N,K)=(64,4096,8192); FP32 output\n"
      "  --kernel all|0|1|2|3|4|5|6|7   0=cuBLAS; 5..7=handwritten Hopper\n"
      "  --bn 32|64|128             tile width (default 64)\n"
      "  --split 1|2|4|8|16|32      K partitions for 4..7 (default 4)\n"
      "  --timing graph|events      default graph; both include reduction\n"
      "  --format table|csv         aligned table by default\n"
      "  --pattern random|ones|alternating   BF16 input (default random)\n"
      "  --iterations INT --samples INT --warmup INT\n"
      "Results go to stdout; metadata and correctness diagnostics to stderr.");
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") { usage(); std::exit(0); }
    if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
    std::string value = argv[++i];
    auto integer = [&] {
      size_t end = 0;
      int v = std::stoi(value, &end);
      if (end != value.size()) throw std::runtime_error("invalid integer: " + value);
      return v;
    };
    if (key == "--kernel") o.kernel = value;
    else if (key == "--timing") o.timing = value;
    else if (key == "--pattern") o.pattern = value;
    else if (key == "--format") o.format = value;
    else if (key == "--bn") o.bn = integer();
    else if (key == "--split") o.split = integer();
    else if (key == "--iterations") o.iterations = integer();
    else if (key == "--samples") o.samples = integer();
    else if (key == "--warmup") o.warmup = integer();
    else throw std::runtime_error("unknown option: " + key);
  }
  if (o.kernel != "all" && o.kernel != "0" && o.kernel != "1" &&
      o.kernel != "2" && o.kernel != "3" && o.kernel != "4" && o.kernel != "5" &&
      o.kernel != "6" && o.kernel != "7")
    throw std::runtime_error("kernel must be all or 0..7");
  if (o.bn != 32 && o.bn != 64 && o.bn != 128)
    throw std::runtime_error("bn must be 32, 64, or 128");
  if (o.split < 1 || o.split > 32 || (o.split & (o.split - 1)))
    throw std::runtime_error("split must be a power of two from 1 to 32");
  if (o.timing != "graph" && o.timing != "events")
    throw std::runtime_error("timing must be graph or events");
  if (o.format != "table" && o.format != "csv")
    throw std::runtime_error("format must be table or csv");
  if (o.pattern != "random" && o.pattern != "ones" && o.pattern != "alternating")
    throw std::runtime_error("pattern must be random, ones, or alternating");
  if (o.iterations < 1 || o.iterations > 10000 || o.samples < 1 || o.samples > 1000 ||
      o.warmup < 1 || o.warmup > 1000)
    throw std::runtime_error("require iterations=1..10000, samples/warmup=1..1000");
  return o;
}

void initialize(std::vector<bf16>& a, std::vector<bf16>& b, const Options& o) {
  std::mt19937 rng(2026);
  std::normal_distribution<float> normal(0.f, 1.f);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      float x = o.pattern == "random" ? normal(rng) : 1.f;
      if (o.pattern == "alternating") x = ((m + k) & 1 ? -1.f : 1.f) * (1.f + (m % 5) * 0.125f);
      a[m * K + k] = __float2bfloat16_rn(x);
    }
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      float x = o.pattern == "random" ? normal(rng) : 1.f;
      if (o.pattern == "alternating") x = ((n + k) & 1 ? -1.f : 1.f) * (1.f + (n % 7) * 0.125f);
      b[n * K + k] = __float2bfloat16_rn(x);
    }
}

// A separate CPU double-precision check catches a shared layout mistake in
// the cuBLAS reference and custom implementations. Full-output GPU comparison
// below still checks every element, including finite values and all partitions.
void verify_reference(const std::vector<bf16>& a, const std::vector<bf16>& b,
                       const std::vector<float>& reference) {
  for (int sample = 0; sample < 32; ++sample) {
    const int m = sample == 31 ? M - 1 : (sample * 17) % M;
    const int n = sample == 31 ? N - 1 : (sample * 131) % N;
    double expected = 0.;
    for (int k = 0; k < K; ++k)
      expected += double(__bfloat162float(a[m * K + k])) * __bfloat162float(b[n * K + k]);
    float actual = reference[m * N + n];
    if (!std::isfinite(actual) || std::abs(actual - expected) > 0.005 + 0.0005 * std::abs(expected))
      throw std::runtime_error("cuBLAS disagrees with CPU at (" + std::to_string(m) + "," + std::to_string(n) + ")");
  }
  std::fprintf(stderr, "cuBLAS reference: 32 CPU-double spot checks passed\n");
}

void verify(const std::vector<float>& actual, const std::vector<float>& reference,
             int kernel) {
  double worst_abs = 0., worst_scaled = 0.;
  for (size_t i = 0; i < ElementsC; ++i) {
    const double err = std::abs(double(actual[i]) - reference[i]);
    const double scaled = err / (0.005 + 0.0005 * std::abs(reference[i]));
    if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) || scaled > 1.) {
      std::fprintf(stderr, "kernel %d: FAIL (%zu,%zu): got %.9g expected %.9g\n",
                   kernel, i / N, i % N, actual[i], reference[i]);
      throw std::runtime_error("correctness check failed");
    }
    worst_abs = std::max(worst_abs, err);
    worst_scaled = std::max(worst_scaled, scaled);
  }
  std::fprintf(stderr, "kernel %d: PASS all %zu outputs; max_abs=%.6g max_tolerance_fraction=%.6g\n",
               kernel, ElementsC, worst_abs, worst_scaled);
}

struct Timing { double median_us, min_us; };

Timing measure(const Launch& run, cudaStream_t stream, const Options& o) {
  // Allocations, descriptor encoding, plan initialization and capture are untimed.
  for (int i = 0; i < o.warmup; ++i) run();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (o.timing == "graph") {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < o.iterations; ++i) run();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    // Upload / first replay overhead must not enter the measurements.
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> times;
  for (int s = 0; s < o.samples; ++s) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (executable) CUDA_CHECK(cudaGraphLaunch(executable, stream));
    else for (int i = 0; i < o.iterations; ++i) run();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(ms * 1000. / o.iterations);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
  if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
  std::sort(times.begin(), times.end());
  const size_t middle = times.size() / 2;
  return {(times[(times.size() - 1) / 2] + times[middle]) / 2., times.front()};
}

template <int BN>
Launch prepare(int kernel, const bf16* a, const bf16* b, float* c,
                float* partial, int split, cudaStream_t stream) {
  switch (kernel) {
    case 1: return [=] { launch_scalar(a, b, c, stream); };
    case 2: return [=] { launch_mma<BN>(a, b, c, stream); };
    case 3: return [=] { launch_async<BN>(a, b, c, 1, stream); };
    case 4: return [=] { launch_split_k<BN>(a, b, c, partial, split, stream); };
    case 5: case 6: case 7:
      return prepare_hopper<BN>(kernel, a, b, c, partial, split, stream);
    default: throw std::runtime_error("unsupported kernel");
  }
}

void print_row(int kernel, const Options& o, Timing t, double baseline) {
  const double tflops = (2. * M * N * K) / (t.median_us * 1.e6);
  if (o.format == "csv") {
    std::printf("%d,%d,%d,%s,%.6f,%.6f,%.6f,%.6f\n", kernel,
        kernel < 2 ? 0 : o.bn, kernel >= 4 ? o.split : 1,
        o.timing.c_str(), t.median_us, t.min_us, tflops, baseline / t.median_us);
  } else {
    std::printf("%-6d  %4d  %5d  %-6s  %12.3f  %12.3f  %12.2f  %11.3fx\n", kernel,
        kernel < 2 ? 0 : o.bn, kernel >= 4 ? o.split : 1,
        o.timing.c_str(), t.median_us, t.min_us, tflops, baseline / t.median_us);
  }
}

int main(int argc, char** argv) try {
  const Options o = parse(argc, argv);
  int device = 0, runtime = 0, driver = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major != 9 || prop.minor != 0)
    throw std::runtime_error("this tutorial targets Hopper SM90 (build uses sm_90a)");
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  std::fprintf(stderr, "GPU=%s SMs=%d L2=%d CUDA_runtime=%d driver=%d\n"
      "M=%d N=%d K=%d A=row B=column C=row BF16->FP32 pattern=%s timing=%s\n"
      "iterations=%d samples=%d warmup=%d seed=2026 cache=steady-state-same-addresses\n",
      prop.name, prop.multiProcessorCount, prop.l2CacheSize, runtime, driver,
      M, N, K, o.pattern.c_str(), o.timing.c_str(), o.iterations, o.samples, o.warmup);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle;
  BLAS_CHECK(cublasCreate(&handle));
  BLAS_CHECK(cublasSetStream(handle, stream));
  BLAS_CHECK(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));

  std::vector<bf16> ha(size_t(M) * K), hb(size_t(N) * K);
  initialize(ha, hb, o);
  std::vector<float> reference(ElementsC), actual(ElementsC);
  DeviceBuffer<bf16> a(ha.size()), b(hb.size());
  DeviceBuffer<float> c(ElementsC), ref(ElementsC), partial(ElementsC * o.split);
  CUDA_CHECK(cudaMemcpyAsync(a.data, ha.data(), ha.size() * sizeof(bf16), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(b.data, hb.data(), hb.size() * sizeof(bf16), cudaMemcpyHostToDevice, stream));
  Launch baseline = [&] { cublas_gemm(handle, a.data, b.data, ref.data); };
  baseline();
  CUDA_CHECK(cudaMemcpyAsync(reference.data(), ref.data, ElementsC * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  verify_reference(ha, hb, reference);
  const Timing base = measure(baseline, stream, o);
  struct Result { int kernel; Timing timing; };
  std::vector<Result> results{{0, base}};

  std::vector<int> kernels;
  if (o.kernel == "all") {
    kernels = {1, 2, 3, 4, 5, 6, 7};
  } else if (o.kernel != "0") kernels.push_back(std::stoi(o.kernel));
  for (int kernel : kernels) {
    Launch run;
    if (o.bn == 32) run = prepare<32>(kernel, a.data, b.data, c.data, partial.data, o.split, stream);
    else if (o.bn == 64) run = prepare<64>(kernel, a.data, b.data, c.data, partial.data, o.split, stream);
    else run = prepare<128>(kernel, a.data, b.data, c.data, partial.data, o.split, stream);
    // NaN poison catches unwritten outputs/partials. No zero initialization can
    // accidentally make a missing store look correct on structured inputs.
    CUDA_CHECK(cudaMemsetAsync(c.data, 0xff, ElementsC * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(partial.data, 0xff, ElementsC * o.split * sizeof(float), stream));
    run();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), c.data, ElementsC * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    verify(actual, reference, kernel);
    const Timing t = measure(run, stream, o);
    // Verify again after repeated replays: catches state that was not reset.
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), c.data, ElementsC * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    verify(actual, reference, kernel);
    results.push_back({kernel, t});
  }
  BLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  // Emit the complete table after all correctness/setup diagnostics, including
  // when stdout and stderr share a terminal or are redirected into one log.
  if (o.format == "csv")
    std::puts("kernel,bn,split,timing,median_us,min_us,tflops,speedup_vs_cublas");
  else
    std::printf("%-6s  %4s  %5s  %-6s  %12s  %12s  %12s  %12s\n",
        "Kernel", "BN", "Split", "Timing", "Median us", "Min us", "TFLOP/s", "Speedup");
  for (const Result& result : results)
    print_row(result.kernel, o, result.timing, base.median_us);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
