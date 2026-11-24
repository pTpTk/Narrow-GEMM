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

These are **user-reported starting results**, before the changes below.
Stage 6's extra machinery did not improve on stage 5 at this configuration.
Stage 7 was about 34% slower than stage 5. The changes below target concrete
overheads. The next user-reported run measures those changes at one configuration.

## Measured follow-up before stages 9–14

The follow-up used BN=64, split=4 for stages 4–8, BK=64 for warp MMA, and
two slots / batch-K=2 for stages 6–8:

| Implementation after the changes | Median us | cuBLAS-relative throughput |
| --- | ---: | ---: |
| cuBLAS | 33.438 | 1.000x |
| Warp MMA, stage 2 | 474.597 | 0.070x |
| Async warp MMA, stage 3 | 122.611 | 0.273x |
| Split-K warp MMA, stage 4 | 66.549 | 0.502x |
| Serial TMA/WGMMA, stage 5 | 49.984 | 0.669x |
| Specialized pipeline, stage 6 | 45.542 | 0.734x |
| Persistent pipeline, stage 7 | 44.525 | 0.751x |
| Cooperative pipeline, stage 8 | 44.672 | 0.749x |

These are user measurements, not local authoring-environment GPU validation.
They show progress but leave an 11.087-us gap between stage 7 and cuBLAS in
this run. They do not establish an optimum over the other candidate settings.

The [source-by-source audit](SOURCE_AUDIT.md) motivated stages 9–14. They now
implement the additional mechanisms below. **No new H200 timings are available**;
the table above remains the last measured checkpoint.

## Next pass: newly implemented optimizations

| Stage | Compare with | Question |
| --- | --- | --- |
| 9 | 7 at the same tile/split/queue settings | Do fused 3D TMA, descriptor prefetch/updates and vector FP32 output reduce overhead? |
| 10 | 9 | Does retaining one outstanding MMA group help? |
| 11 | 10 | Does a TMA output pipeline offset its output-buffer footprint? |
| 12 | 10 | Does Stream-K balancing/fixup beat uniform split-K plus full reduction? |
| 13 | 10 | Do two-CTA A multicast savings offset cluster synchronization and placement? |
| 14 | 11 | Can alternating consumer WGs overlap whole-tile math and epilogues efficiently? |

```bash
make -B
python3 scripts/sweep.py --profile advanced --kernels 5,7,8,9,10,11,12,13,14 \
  --csv ../h200-advanced.csv 2> ../h200-advanced.log
```

The advanced profile has 110 processes including its reference. It covers
BN=16/32/64/128/256, splits 1/2/4, deeper rings through six merged K128 slots,
eight K64 slots, worker caps, and K64 ping-pong configurations. Candidate
settings are offline hypotheses. Invalid shared-memory/cluster configurations
appear as explicit skips; they are retained in CSV and the final aligned table.
Actual shared capacity, occupancy, SM count and cluster residency are queried.
Stage 14 intentionally reserves over half the per-SM shared capacity, allowing
one resident CTA and enough registers for the consumer budgets. That can lose
against the 160-thread paths on a memory-bound problem.

In stage 12 only, `--split` caps the worker budget at `Ntiles*split`; it does
not require that many contributors per tile. Fully owned tiles write C directly;
boundary tiles publish FP32 partials and the last arrival reduces them in the
same kernel. Two boundary slots per worker are allocated. This allocation need
not be smaller than uniform split-K workspace, even when partial traffic drops.
All fixup/counters are timed. Counters reset within each complete invocation;
the plan owns its workspace and requires sequential use on its captured stream.

Start with stage 10 BN=64/split=2 at two and six slots. At BN=64/batch-K=2,
six input slots use 197,632 bytes; adding FP32 TMA output uses 214,016 bytes.
These configurations normally allow only one resident CTA per SM. Compare
their latency hiding against the smaller two-slot grid rather than assuming
that deeper buffering wins.

## Earlier changes, measured above

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
the updated warp-MMA checkpoints and the BN=64 stage-9–14 candidates. Its parameters are offline hypotheses, not
autotuned defaults or a claim of optimality. The table and CSV include `tile_k`,
`stages`, and `workers_per_sm`; zero in the last column means automatic for
stage 7 and stages 9–13; stage 14 reports its enforced cap of one. Actual residency/worker count is in stderr.

The original 33.499-us reference gives a scale for interpreting results:
0.9x would require about 37.22 us and 1.0x would require 33.50 us. Re-measure
cuBLAS rather than treating those times as fixed across runs or clock states.
All partial GEMM and FP32 reduction/fixup work remains inside the timed operation.

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

For new stages, also exercise non-power-of-two queue depths and many replays:

```bash
../build/narrow_gemm --kernel 12 --bn 256 --split 32 --stages 3 --batch-k 2 \
  --pattern alternating --iterations 100 --samples 10
../build/narrow_gemm --kernel 13 --bn 64 --split 32 --stages 3 --batch-k 2 \
  --pattern ones --iterations 100 --samples 10
../build/narrow_gemm --kernel 14 --bn 64 --split 32 --stages 3 --batch-k 1 \
  --pattern random --iterations 100 --samples 10
compute-sanitizer --tool racecheck ../build/narrow_gemm --kernel 11 \
  --split 32 --stages 3 --iterations 2 --samples 1 --timing events
compute-sanitizer --tool synccheck ../build/narrow_gemm --kernel 14 \
  --split 4 --batch-k 1 --iterations 2 --samples 1 --timing events
```

Run memcheck and racecheck on 12–14 too. A cluster `SKIP` on a particular MIG
configuration is a supported outcome; a numerical mismatch or CUDA error is not.
