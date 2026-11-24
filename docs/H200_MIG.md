# Offline candidates for H200 3g.71gb

The supplied measurement used BF16 inputs, FP32 output, graph timing,
BN=64 and split=4 for stages 4–7:

| Implementation before these changes | Median us | cuBLAS-relative throughput |
| --- | ---: | ---: |
| cuBLAS | 33.499 | 1.000x |
| Warp MMA, stage 2 | 514.248 | 0.065x |
| Async warp MMA, stage 3 | 203.190 | 0.165x |
| Split-K warp MMA, stage 4 | 83.469 | 0.401x |
| Serial TMA/WGMMA, stage 5 | 49.978 | 0.670x |
| Specialized pipeline, stage 6 | 50.595 | 0.662x |
| Persistent pipeline, stage 7 | 67.086 | 0.499x |

These are **user-reported baseline results**, not measurements of the new code.
Stage 6's extra machinery did not improve on stage 5 at this configuration.
Stage 7 was about 34% slower than stage 5. The changes below target concrete
overheads; none is a measured winner yet.

## Changes to test

- **Keep stage 5 as the reference.** Its serial kernel and accumulator path are
  retained, so the first sweep can reproduce the known configuration.
- **Stages 2–4: larger K tile.** BK=64 halves the outer loop's copy/barrier
  iterations versus BK=32. `make -B WARP_BK=32`, `64`, or `128` allows a separate
  build comparison. Larger tiles cost shared memory; more is not always faster.
  Dynamic shared storage and its launch attribute are configured before timing.
- **Stage 6: 160 threads instead of 256.** One producer warp is enough for a
  single elected TMA issuer, alongside the 128-thread WGMMA consumer group.
  Consumer warps release an operand slot with four arrivals, down from 128.
  Each warp arrives only after completing its WGMMA wait.
- **Stages 6–8: batch K chunks.** `--batch-k 2` combines two existing 64-wide
  TMA boxes under one completion barrier and one eight-instruction MMA group.
  This halves completion/release rounds per K interval. It does not enlarge
  the 128-byte TMA swizzle box to an invalid 256-byte box. Compare with
  `--batch-k 1`; each independently swizzled chunk has its own shared address.
- **Stages 6–8: uniform accumulation.** Initialize accumulators before the
  K loop and use a uniform chain of accumulating MMA instructions. This costs
  a zero-fill but removes a scale-D branch inside the asynchronous group.
  The CUDA 12.6.85 compiler no longer reports the earlier serialization
  diagnostics for these tuned paths; stage 5 keeps its original path.
- **Stage 7: occupancy-sized workers.** The old code launched at most one
  persistent CTA per SM and forced a 128-register consumer budget. The new
  code uses normal register allocation and queries legal CTA residency with
  the actual kernel's threads/shared memory. The worker grid can then cover
  several CTAs per visible SM. The TMA paths prefer the maximum shared-memory
  carveout because their input transfers bypass L1. `--workers-per-sm 1` isolates the smaller-grid
  policy; it does not restore the old 256-thread/register-redistribution code.
- **New stage 8: cooperative prefetch.** A 128-thread warpgroup issues future
  TMA loads and computes the current tile. There are no dedicated producer
  threads or empty-slot arrivals. A block barrier after MMA completion protects
  a consumed buffer before its next refill. This is a useful alternative when
  a separate producer adds more scheduling cost than latency hiding.

The default candidate uses two ring slots, each containing two K=64 chunks.
That is the same input capacity as four slots containing one chunk, with half
as many completion rounds. At BN=64 it allocates 66,560 shared bytes, including
alignment/barriers. Two slots with one chunk allocate 33,792 bytes; four slots
with one chunk also allocate 66,560 bytes. These are distinct tradeoffs between
residency and latency hiding. The sweep includes them as separate rows.

The visible device's SM count and L2 capacity matter. Do not apply the full
H100 tutorial's 132-SM wave model or full-chip bandwidth to H200 MIG, and do not
derive its memory bandwidth by multiplying by 3/7. Compute and memory resources
are partitioned separately. The code queries SM count and occupancy at runtime;
no assumed 3g SM count is built in.

## First measurement pass

Rebuild, then compare stage 5 and the new cooperative pipeline first:

```bash
make -B
python3 scripts/sweep.py --profile h200-mig --kernels 5,8 \
  --csv ../h200-first.csv 2> ../h200-first.log
```

This is 24 benchmark processes including the initial reference. It emphasizes
BN=64/128, splits 1/2/4, plus a few BN=32 cases and buffering alternatives.
All candidate rows compare against their own process's fresh cuBLAS baseline.
Then include the specialized and persistent paths:

```bash
python3 scripts/sweep.py --profile h200-mig --kernels 5,6,7,8 \
  --csv ../h200-pipelines.csv 2> ../h200-pipelines.log
```

This is 46 processes. `--profile h200-mig` without a kernel filter also includes
the updated warp-MMA checkpoints. Its parameters are offline hypotheses, not
autotuned defaults or a claim of optimality. The table and CSV include `tile_k`,
`stages`, and `workers_per_sm`; zero in the last column means automatic for
stage 7 and is unused by other stages. Actual residency/worker count is in stderr.

The original 33.499-us reference gives a scale for interpreting results:
0.9x would require about 37.22 us and 1.0x would require 33.50 us. Re-measure
cuBLAS rather than treating those times as fixed across runs or clock states.
Both partial GEMM and FP32 reduction remain inside the timed operation.

For an individual candidate:

```bash
../build/narrow_gemm --kernel 8 --bn 64 --split 2 --stages 2 --batch-k 2
../build/narrow_gemm --kernel 6 --bn 128 --split 2 --stages 2 --batch-k 1
../build/narrow_gemm --kernel 7 --bn 64 --split 4 --workers-per-sm 1
../build/narrow_gemm --kernel 7 --bn 64 --split 4 --workers-per-sm 0
```

## Correctness before ranking

Run random, ones and alternating inputs. Exercise the shortest K interval with
four slots, where a persistent job can end partway through the ring:

```bash
../build/narrow_gemm --kernel 7 --split 32 --stages 4 --batch-k 2 \
  --pattern ones --iterations 1 --samples 1 --timing events
compute-sanitizer --tool racecheck ../build/narrow_gemm --kernel 8 \
  --split 32 --stages 4 --batch-k 2 --iterations 1 --samples 1 --timing events
```

The authoring environment has no GPU. Compilation, register reports and CPU
schedule audits are useful offline checks but cannot validate hardware TMA
completion, numerical results, race freedom, occupancy behavior or speedup.
Keep the full-output GPU comparisons and sanitizer checks enabled when ranking
these candidates on H200.
