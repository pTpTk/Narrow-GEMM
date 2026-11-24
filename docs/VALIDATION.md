# Validation record

The authoring environment has no GPU. CUDA tools, reference checkouts, compiler
logs, object/PTX files and CPU/mock audits live in the parent workspace, outside
this repository. The H200 3g.71gb baseline in [H200_MIG.md](H200_MIG.md) was
reported by the user; the changed kernels have no measured GPU speedup yet.

## Offline checks completed

- CUDA 12.6.85 compile-only validation passed for the complete default BK=64
  build: 47 local kernel instantiations, covering BN=32/64/128 and every
  stages=2/4, batch-K=1/2 specialization. All 47 reports are spill-free.
- Separate BK=32 and BK=128 warp-kernel probes compiled all six early-kernel
  variants per build, without spills. Dynamic shared memory supports the
  configurations above the static 48-KiB limit.
- Compiler flags included `-std=c++17 -O3 -DNDEBUG -DNARROW_WARP_BK=64
  -gencode=arch=compute_90a,code=sm_90a -Xptxas=-v`.
- The tuned stage-6/7/8 paths emit no WGMMA serialization diagnostics in this
  compiler. The retained stage-5 reference still emits C7519 fence-insertion
  and C7515 serialization messages. These are performance diagnostics, not
  compilation failures; their absence is not a measured speedup.
- An emitted-PTX audit found 47 local kernel entries, no device function calls,
  no library-kernel dependencies and no forced `setmaxnreg` instructions.
  CUTLASS/CuTe/DeepGEMM headers are absent from the build include path. cuBLAS
  remains isolated in the benchmark reference.
- The actual C++ option parser and row formatter were exercised in a CPU-only
  harness: each tuning option is accepted, invalid values rejected, and stage-5
  metadata remains K=64/one slot regardless of unrelated pipeline flags.
- The H200-MIG sweep was checked with an explicitly labeled mock executable:
  24 cases for kernels 5/8, 46 for 5/6/7/8, and 54 for the complete profile.
  All buffering/worker dimensions survive table and CSV output. Diagnostics
  precede the table; completed cases survive a simulated failure.
- CPU state-machine audits exercised 576 randomized producer/TMA/consumer
  schedules, 24 cooperative schedules, persistent job boundaries, exact K/job
  coverage and per-box swizzle address bijections. These check schedule/layout
  invariants, not CUDA execution or numerical correctness.
- The generic SM90 target rejection, Makefile command generation, documentation
  paths and absence of scratch artifacts from this repository were checked.

The current code has **not been linked against a complete cuBLAS runtime or
executed on H200/H100 here**. Offline checks cannot establish descriptor
acceptance, numerical correctness, race freedom, actual residency or speed.

## Default compiler resources

These are current ptxas register counts and host-computed launch storage sizes,
not GPU measurements. Stages 6–8 use two slots and two K chunks per slot;
stages 2–3 use BK=64. All shared allocations below are dynamic.

| Stage | BN | Registers / thread | Shared bytes / CTA | Spill loads/stores |
| --- | ---: | ---: | ---: | ---: |
| 2 | 32 | 56 | 13,824 | 0 / 0 |
| 2 | 64 | 64 | 18,432 | 0 / 0 |
| 2 | 128 | 124 | 27,648 | 0 / 0 |
| 3 | 32 | 64 | 27,648 | 0 / 0 |
| 3 | 64 | 78 | 36,864 | 0 / 0 |
| 3 | 128 | 125 | 55,296 | 0 / 0 |
| 5 | 32 | 42 | 13,312 | 0 / 0 |
| 5 | 64 | 58 | 17,408 | 0 / 0 |
| 5 | 128 | 90 | 25,600 | 0 / 0 |
| 6 | 32 | 42 | 50,176 | 0 / 0 |
| 6 | 64 | 58 | 66,560 | 0 / 0 |
| 6 | 128 | 90 | 99,328 | 0 / 0 |
| 7 | 32 | 42 | 50,176 | 0 / 0 |
| 7 | 64 | 58 | 66,560 | 0 / 0 |
| 7 | 128 | 90 | 99,328 | 0 / 0 |
| 8 | 32 | 42 | 50,176 | 0 / 0 |
| 8 | 64 | 58 | 66,560 | 0 / 0 |
| 8 | 128 | 90 | 99,328 | 0 / 0 |

The tuned specialized/persistent kernels use 160 threads; cooperative stage 8
and reference stage 5 use 128. Register/shared allocation granularity and other
hardware limits affect residency. The launch setup queries CUDA occupancy and
prints the actual shared size, legal resident CTAs, visible SMs and worker grid;
resource counts alone are not throughput predictions.

## Required checks on H200

1. Build and run random, ones and alternating inputs for every candidate.
2. Check full-output comparisons before and after graph replay. Reduction
   remains timed; its summation and FP32 output contract are unchanged.
3. Exercise split=32, stages=4, batch-K=2: persistent jobs can end midway through
   a ring traversal. Run memcheck and racecheck, especially stages 6–8.
4. Compare stage 5 against the reported baseline before ranking changed paths.
   Then run the focused H200-MIG sweep and retain its local cuBLAS comparisons.
5. Investigate the winning candidates with Nsight Compute. Use unprofiled times
   for speedups and actual HBM/L2 traffic when interpreting memory limits.

## Interpretation limits

- Baseline timings in the H200 note are user-reported. There are no timings for
  the changed implementations in this validation record.
- cuBLAS uses GemmEx, not an exhaustive cuBLASLt algorithm search.
- Inputs are BF16; accumulators, partials and output are FP32.
- Timing is repeated-address steady state, not guaranteed cold-cache latency.
- Tensor-map encoding, occupancy queries and graph construction are untimed.
- One aligned shape is supported. Dimension/layout changes need a fresh audit
  of bounds, fragments, descriptors, shared alignment, reduction and workspace.
