# Validation record

The authoring environment has no GPU. CUDA tools, reference checkouts, compiler
logs, object/PTX files and CPU/mock audits live in the parent workspace, outside
this repository. Starting and follow-up H200 3g.71gb measurements in
[H200_MIG.md](H200_MIG.md) were reported by the user; no local GPU validation
was performed. The [source audit](SOURCE_AUDIT.md) maps the implemented follow-up
and scope limits.

## Offline checks completed

- CUDA 12.6.85 compile-only validation passed for the complete default BK=64
  build: **87 local kernel instantiations, all spill-free**. These include the
  47 legacy specializations (BN=32/64/128, stages=2/4, batch-K=1/2), plus 40
  advanced variants (BN=16/32/64/128/256, batch-K=1/2; separate drain/overlap,
  multicast and ping-pong kernels). Advanced slot counts are runtime values.
- Separate BK=32 and BK=128 warp-kernel probes compiled all six early-kernel
  variants per build, without spills. Dynamic shared memory supports the
  configurations above the static 48-KiB limit.
- Compiler flags included `-std=c++17 -O3 -DNDEBUG -DNARROW_WARP_BK=64
  -gencode=arch=compute_90a,code=sm_90a -Xptxas=-v`.
- The tuned stage-6/7/8 and new stage-9–14 paths emit no WGMMA serialization
  diagnostics in this compiler. The retained stage-5 reference still emits C7519 fence-insertion
  and C7515 serialization messages. These are performance diagnostics, not
  compilation failures; their absence is not a measured speedup.
- An emitted-PTX audit found 87 local kernel entries, no device function calls
  or library-kernel dependencies. All 40 advanced variants emit descriptor
  prefetch and 3D input TMA. Overlap variants emit wait1 and tail wait0; the
  drain-only baseline has no wait1. Output maps/shared vector stores/TMA store
  groups, acquire-release counters and cache-bypassing partial loads appear in
  the advanced engine. Cluster variants emit multicast, remote arrivals and
  required cluster dimensions; ping-pong variants emit full-WG register budgets.
  Stage 14 is the only path enabling `setmaxnreg`.
  CUTLASS/CuTe/DeepGEMM headers are absent from the build include path. cuBLAS
  remains isolated in the benchmark reference.
- The actual C++ option parser and row formatter were extracted into a
  CPU-only harness: 450 kernel/BN/depth combinations passed; invalid values
  were rejected; stage-5 metadata stays K64/one slot. CSV status/decomposition,
  enforced stage-14 cap and aligned skipped rows were checked. After timed
  replay validation, the benchmark also re-poisons C/partials and runs the same
  plan once more, without resetting its internal counters; stale correct outputs
  cannot mask a missing write in that check. This GPU check is implemented but
  has not been executed here.
- The previous H200-MIG profile was checked with an explicitly labeled mock:
  24 cases for kernels 5/8, 46 for 5/6/7/8, and 54 before adding advanced stages.
  A new 110-case advanced-profile mock run exercises all new stages, deep
  queues, BN extremes and explicit exit-77 skips. Failure injection still
  produces a completed-case table and an unsuccessful exit. These are mock
  harness checks, not GPU timings. All buffering/worker dimensions survive
  table and CSV output. Diagnostics precede the table; completed cases survive a simulated failure.
- CPU state-machine audits exercised 576 randomized producer/TMA/consumer
  schedules, 24 cooperative schedules, persistent job boundaries, exact K/job
  coverage and per-box swizzle address bijections. These check schedule/layout
  invariants, not CUDA execution or numerical correctness.
- The reproducible [check_schedules.py](../scripts/check_schedules.py) models
  480 advanced ring/multicast schedules with arbitrary copy/MMA/warp-release
  interleavings, short segments and cross-job wrapping; 3,421 Stream-K
  partitions with three randomized-finisher replays each; and 180 independent
  ping-pong schedules with asynchronous output reads/writes. It checks exact
  coverage, bounded partial slots, last-arrival fixup, deterministic owner
  order, counter reset, output reuse and FP32 pair alignment. These models
  abstract GPU memory ordering rather than proving it.
- A CPU harness using the actual extracted C++ WorkIterator/AdvancedStorage
  checked ten BN/slab variants, 257 Stream-K worker counts per variant, every
  uniform split, and every runtime queue depth. This cross-checks the Python
  models against the implementation's interval and shared-offset formulas.
- The generic SM90 target rejection, Makefile command generation, documentation
  paths and absence of scratch artifacts from this repository were checked.

The current code has **not been linked against a complete cuBLAS runtime or
executed on H200/H100 here**. Offline checks cannot establish descriptor
acceptance, numerical correctness, race freedom, actual residency or speed.

## Compiler resources for stages 1–8

These are retained ptxas register counts and host-computed launch storage sizes,
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

## Advanced compiler resources

For batch-K=2, the advanced drain and overlap engines have the same reported
register count; modes 9–12 share that engine except for the wait-policy
specialization. Queue depth is runtime and does not create extra compile
specializations. There are no spills in either batch-K=1 or 2.

| BN | Stages 9–12 regs/thread | Stage 13 regs/thread | Stage 14 static regs/thread |
| --- | ---: | ---: | ---: |
| 16 | 37 | 36 | 128 |
| 32 | 48 | 43 | 128 |
| 64 | 64 | 58 | 128 |
| 128 | 96 | 96 | 128 |
| 256 | 178 | 167 | 168 |

Stage 14 requests consumer budgets of 128 (192 for BN256) after the full
producer WG releases to 40. Its host setup enforces one CTA/SM through shared
capacity reservation. Some compiled configurations cannot fit shared memory
and are skipped at runtime; a compiled resource report does not imply a legal
launch. In particular BN256 does not fit stage 14's two output buffers plus
two input rings at any exposed depth/slab setting.

At BN=64, two K128 slots request 66,560 bytes for 9/10/12/13 and 82,944 for
11. Six slots request 197,632 and 214,016 respectively. Stage 14 with two K64
slots needs 99,328 bytes before the one-CTA reservation, whose final amount is
derived from `sharedMemPerMultiprocessor`. Actual allocations and residency
are logged at setup on the target device.

## Required checks on H200

1. Build and run random, ones and alternating inputs for every candidate.
2. Check full-output comparisons before and after graph replay. Reduction
   and Stream-K fixup remain timed; the FP32 output contract is unchanged.
3. Exercise split=32, stages=4, batch-K=2: persistent jobs can end midway through
   a ring traversal. Also exercise non-power-of-two depths, one-group Stream-K
   segments, odd Stream-K worker counts, cluster lifetime and ping-pong output
   reuse. Run memcheck, racecheck and synccheck on stages 6–14; see the commands
   in [H200_MIG.md](H200_MIG.md#correctness-before-ranking).
4. Compare stage 5 against the reported baseline before ranking changed paths.
   Then run the advanced sweep and retain its local cuBLAS comparisons.
5. Investigate the winning candidates with Nsight Compute. Use unprofiled times
   for speedups and actual HBM/L2 traffic when interpreting memory limits.

## Interpretation limits

- Starting and follow-up timings are user-reported in the H200 note; the
  compiler/resource table in this file is not a GPU timing measurement.
- cuBLAS uses GemmEx, not an exhaustive cuBLASLt algorithm search.
- Inputs are BF16; accumulators, partials and output are FP32.
- Timing is repeated-address steady state, not guaranteed cold-cache latency.
- Tensor-map encoding, occupancy queries and graph construction are untimed.
- One aligned shape is supported. Dimension/layout changes need a fresh audit
  of bounds, fragments, descriptors, shared alignment, reduction and workspace.
