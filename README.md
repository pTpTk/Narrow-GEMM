# Narrow GEMM on H100: a worklog you can run

A step-by-step tutorial for **one** dense GEMM: **`(M,N,K) = (64,4096,8192)`**,
BF16 inputs, FP32 accumulation and output. Start with the [tutorial](TUTORIAL.md).
It follows the numbered experiments and inspect-the-bottleneck structure of
[Pranjal Shankhdhar's H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog),
with original explanations focused on narrow matrices.

The interesting constraint is the number of independent output tiles. A fast
square-GEMM tile can leave most of an H100 idle here. We build up Tensor Core
execution, overlapping copies, split-K, then a handwritten Hopper pipeline.
[CUTLASS and DeepGEMM source notes](docs/REFERENCES.md) identify exactly which
upstream ideas inform each experiment.

| Stage | File | What changes |
| --- | --- | --- |
| 0 | [cuBLAS baseline](include/reference.cuh) | Reference and timed comparison only |
| 1 | [Scalar CUDA](kernels/01_scalar.cuh) | One output per thread |
| 2 | [Warp MMA](kernels/02_mma.cuh) | Explicit `ldmatrix` / `mma.sync`, vector copies and reuse |
| 3 | [Asynchronous copies](kernels/03_async.cuh) | Two shared-memory buffers and `cp.async` |
| 4 | [Split-K](kernels/04_split_k.cuh) | More CTAs, FP32 partials, deterministic reduction |
| 5 | [TMA/WGMMA](kernels/05_tma_wgmma.cuh) | Hand-encoded layouts, raw TMA and WGMMA PTX |
| 6 | [Warp specialization](kernels/06_warp_specialized.cuh) | Producer/consumer warpgroups, a three-slot ring and transaction barriers |
| 7 | [Persistent workers](kernels/07_persistent.cuh) | Explicit job loop, persistent ring phases and register redistribution |

All seven custom stages implement their math and scheduling from scratch.
They include no CUTLASS, CuTe, DeepGEMM, WMMA, or CUDA barrier/pipeline helper
headers. [hopper_ptx.cuh](kernels/hopper_ptx.cuh) and
[wgmma.cuh](kernels/wgmma.cuh) contain our instruction wrappers, descriptor bits,
register lists and output mapping. CUTLASS/DeepGEMM are reading references only.
CUDA runtime/driver APIs provide allocation, launch and tensor-map encoding;
cuBLAS is called only by the benchmark reference. No GEMM work is delegated.

## Build and run

Use Linux, an H100, a compatible driver, and **CUDA Toolkit 12.6** with cuBLAS.
The target is `sm_90a`; this is deliberately not a portable GPU library. CUDA
12.6.85 compile checks were performed for all stages; see
[validation status](docs/VALIDATION.md). **GPU execution and performance have not
been measured in the authoring environment.**

From this directory:

```bash
make
../build/narrow_gemm --kernel all
../build/narrow_gemm --kernel 4 --bn 64 --split 4
../build/narrow_gemm --kernel 4 --bn 64 --split 4 --pattern ones
../build/narrow_gemm --kernel 4 --bn 64 --split 4 --pattern alternating
```

The same binary contains the handwritten Hopper stages:

```bash
../build/narrow_gemm --kernel 5 --bn 64 --split 4
../build/narrow_gemm --kernel 6 --bn 64 --split 4
../build/narrow_gemm --kernel 7 --bn 64 --split 4
```

No reference-library checkout or Python package is needed.
Use `make NVCC=/path/to/cuda/bin/nvcc` if needed. Add
`NVCCFLAGS=-Xptxas=-v` to inspect registers, shared memory and spills.

If ptxas reports that WGMMA is unsupported on `.target sm_90`, the actual build
selected generic Hopper instead of `sm_90a`. Rebuild with `make -B`; the
Makefile uses `-gencode=arch=compute_90a,code=sm_90a`. For a custom build or
fast.cu wrapper, use that same explicit target. CUDA 12.6's `-arch=sm_90a`
shorthand also emits generic `compute_90` PTX, so avoid the shorthand for this
code. Remove any additional generic `sm_90`/`compute_90` targets: every device
target must support the inline PTX, even when you only intend to run an earlier
kernel stage.
`__CUDA_ARCH__` is 900 for both variants, so the code also checks the
architecture-specific feature macro and reports a direct build error.

## Collect your own worklog

```bash
python3 scripts/model.py --sms 132 > ../model.txt
python3 scripts/sweep.py --csv ../results.csv 2> ../run.log
# For CSV on stdout instead of the readable table:
python3 scripts/sweep.py --format csv > ../results.csv 2> ../run.log
```

The sweep tests `BN={32,64,128}`, with `S={1,2,4,8,16}` for stages 5–7
and `S={2,4,8,16}` for stage 4. The scalar stage is optional (`--include-scalar`)
because it is intentionally slow. Progress, metadata and correctness diagnostics
go to stderr. Once all cases finish, stdout prints one aligned table grouped
by kernel, tile width and split. Each row includes its own run's cuBLAS time and
speedup; repeated baseline rows are omitted from this readable summary.
`--csv PATH` also saves full-precision records with every local baseline, and
`--format csv` selects that format for stdout. The benchmark binary itself also
emits its CSV rows together after all checks. On failure the sweep prints only
completed cases and exits unsuccessfully. Keep CSV, logs, downloaded libraries,
build output and profiler reports in the parent workspace.

`--timing graph` is the default: capture a batch of complete GEMMs and time its
replay with CUDA events. `--timing events` times ordinary stream submissions.
Both include split-K reduction; neither includes allocation, input generation,
plan initialization, correctness checks or graph construction. Times are
steady-state repeated-address measurements, not guaranteed cold-HBM latency.

For correctness and race checking on the GPU:

```bash
compute-sanitizer --tool memcheck ../build/narrow_gemm --kernel all \
  --iterations 1 --samples 1 --timing events
compute-sanitizer --tool racecheck ../build/narrow_gemm --kernel 4 \
  --iterations 1 --samples 1 --timing events
compute-sanitizer --tool memcheck ../build/narrow_gemm --kernel 7 \
  --iterations 1 --samples 1 --timing events
compute-sanitizer --tool racecheck ../build/narrow_gemm --kernel 6 \
  --iterations 1 --samples 1 --timing events
```

These are instructions to run, not claims of completed GPU validation. The
[validation notes](docs/VALIDATION.md) record the checks actually completed.

## Reading map

- [Tutorial: hypotheses, kernels, and what to measure](TUTORIAL.md)
- [Pinned upstream sources and precise connections](docs/REFERENCES.md)
- [Validation status and benchmark limitations](docs/VALIDATION.md)
- [Benchmark driver](bench.cu), [shared tile mechanics](kernels/tile.cuh)
- [Analytical model](scripts/model.py), [experiment sweep](scripts/sweep.py)
