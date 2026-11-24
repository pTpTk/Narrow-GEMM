# Source audit and implemented follow-up

Audited on **2026-11-24** against pinned SM90 dense-BF16 paths in CUTLASS
v3.9.2, DeepGEMM and fast.cu stages 4–12. Revisions are in
[REFERENCES.md](REFERENCES.md). The audit initially identified gaps in stages
1–8; **stages 9–14 now implement the additional applicable mechanisms**.
All implementations are handwritten CUDA/PTX. Upstream headers are not included.
This is a scoped comparison of these paths, not every feature of each library.

## Measured checkpoint

The latest user-reported run used graph timing, BN=64, split=4 for stages 4–8,
and two slots / two K chunks for stages 6–8:

| Kernel | Median us | Relative throughput to cuBLAS |
| --- | ---: | ---: |
| cuBLAS | 33.438 | 1.000x |
| Serial stage 5 | 49.984 | 0.669x |
| Specialized stage 6 | 45.542 | 0.734x |
| Persistent stage 7 | 44.525 | 0.751x |
| Cooperative stage 8 | 44.672 | 0.749x |

There are **no hardware correctness or performance results for stages 9–14**
in the authoring environment. Their compilation and CPU-model checks are
recorded in [VALIDATION.md](VALIDATION.md). The measured 11.087-us gap does not
establish which mechanism limits performance or which new candidate will win.

## Coverage matrix

“Implemented” means an executable path exists; it does not mean GPU-validated,
measured faster, or identical to an upstream scheduler/heuristic.

| Technique | Exact upstream evidence | Local implementation | Shape-specific consequence |
| --- | --- | --- | --- |
| BF16 WGMMA, swizzled input descriptors | [DeepGEMM][D-desc], [fast.cu][F-core] | 5–14, direct PTX and explicit register tuples | M=64 throughout; no padded M=128 math |
| TMA producer/consumer queues | [DeepGEMM][D-prod] | 6/7 producer warp, 8 cooperative, 9–14 advanced rings | Input lifetime ends only after completed MMA reads |
| Warp-level empty arrivals | [DeepGEMM][D-loop] | Four per local consumer WG; 13 uses eight from a two-CTA cluster | Avoids per-lane barrier arrivals |
| K-stage merging | [DeepGEMM][D-merge] | Batch-K=1/2; one commit per K64/K128 group | Amortizes barriers and MMA waits |
| Outstanding committed MMA groups | [CUTLASS][C-mma] | 10–14, `compute_advanced`: separate read/release cursors, `wait_group 1`, tail `wait_group 0` | Never releases the newest outstanding operand slot |
| Tensor-map descriptor prefetch | [DeepGEMM][D-prefetch], [CUTLASS][C-prefetch] | 9–14, raw `prefetch.tensormap`; output map in 11/14 | Untimed map encoding; device prefetch remains timed |
| Descriptor low-word updates/uniformization | [DeepGEMM][D-desc], [consumer][D-loop] | 9–14, fixed high word, low-word `shfl`, K16 descriptor increments | Shared bases retain 1024-byte alignment |
| Wider N candidates/deeper queues | [DeepGEMM candidates][D-candidates], [budget][D-budget] | 9–14: BN=16/32/64/128/256; slots=2/3/4/5/6/8; runtime capacity check | Every N tile divides 4096; legal combinations are still device/resource dependent |
| Fused multidimensional TMA input | [fast.cu][F-3d] | 9–14, `[64,rows,K/64]` maps and one 3D copy per operand | K128 uses two TMA instructions per group instead of four |
| First MMA scale-D=0 | [CUTLASS][C-mma], [fast.cu][F-zero] | 5 and 9–14; explicit first-group prologue in advanced paths | No accumulator clear in the advanced K loop |
| Aligned FP32 vector output | [DeepGEMM FP32 branch][D-epilogue], [fast.cu ordering][F-zero] | 9/10/12/13 direct `st.global.v2.f32`; 11/14 `st.shared.v2.f32` | Retains FP32 output/partials; normal write-back caching suits immediate partial reduction |
| Asynchronous TMA FP32 epilogue | [DeepGEMM][D-epilogue], [fast.cu][F-epilogue] | 11/14: shared buffer, proxy fences, named barriers, async store, wait-read before reuse, final wait-all | Output storage trades against input depth and CTA residency |
| Stream-K and reduction fixup | [CUTLASS][C-streamk] | 12: contiguous tile/K-group worker intervals; boundary partials; last-arrival acquire/release counter; ordered fixup in same kernel | No spin on another CTA; no separate full reduction launch; locally specialized policy |
| Cluster A multicast | [DeepGEMM][D-prod], [fast.cu][F-cluster] | 13: cluster(2,1,1), N-neighbor A multicast, per-CTA B; arm handshake and remote reader release | Shares A; B has no neighboring M tile to share with |
| Consumer ping-pong | [CUTLASS][C-pingpong] | 14: two consumer WGs alternate whole tiles, independent input rings and output buffers | One WG's MMA can overlap the other's epilogue |
| Dynamic register redistribution | [DeepGEMM][D-registers], [fast.cu][F-registers] | 14: full producer WG releases to 40; consumers request 128 (192 for BN256) | 384-thread CTA; one CTA/SM resource reservation; distinct from the 160-thread producer-warp paths |
| Persistent workers | [DeepGEMM scheduler][D-scheduler], [host][D-host], [fast.cu][F-persistent] | 7 and 9–14 retain ring phases across jobs; device/occupancy-sized workers; cluster occupancy in 13 | DeepGEMM's SM-count grid and candidate scoring are different policies |
| 2D locality/Hilbert traversal | [DeepGEMM][D-swizzle], [fast.cu][F-hilbert] | Linear N order within a K slice; no 2D Hilbert implementation | One row of M tiles leaves no two-axis output traversal to exploit |
| BF16 STSM/stmatrix epilogue | [DeepGEMM BF16 branch][D-bf16-store], [fast.cu][F-stmatrix] | Outside FP32 output contract | BF16 conversion would change benchmark accuracy and output traffic |
| FP8/scales, grouped/ragged GEMM, SM100 instructions | Other paths outside audit | Outside scope | Different arithmetic, workload or hardware |

## Critical protocol differences

### Outstanding groups follow CUTLASS, not the inspected DeepGEMM BF16 loop

CUTLASS sets `K_PIPE_MMAS=1`, issues the next group, waits until at most one
remains, then releases the **older** input slot. Its tail drains and releases
the last slot. Stages 10–14 reproduce that lifetime pattern. Stage 9 keeps the
drain-after-each-group path as a comparison. Read/release ticks persist across
jobs and wrap with runtime modulo, so non-power-of-two queues work too.

The inspected DeepGEMM BF16 loop uses `wait<0>` after each merged stage; its
K merging and resource policy must not be confused with CUTLASS's outstanding
group pipeline. The advanced first group is a separate scale-D=0 prologue.
Drain and overlap variants compile separately: a runtime branch between waits
caused ptxas to serialize MMA and was removed. The resulting new paths emit
no WGMMA serialization diagnostics in CUDA 12.6.85; SASS/hardware still decides
the practical overlap.

### Input packaging and buffering are separate choices

fast.cu uses input dimensions `[64,rows,K/64]`, byte strides `[2*K,128]`, box
`[64,tile_rows,BatchK]` and 128-byte swizzling. Advanced stages use that mapping
without repacking global inputs. Batch-K=2 issues one A and one B copy, whereas
stages 6–8 issue two of each. The pinned fast.cu default is BK=64; wider slabs
are template capability, not proof its published run used K128.

DeepGEMM budgets shared output/barriers before input slots. For BM=BN=64,
FP32 output and its no-scale formulas: 232448-byte budget minus 16384 output
and 256 barrier bytes leaves room for 13 K64 stages, merged into six K128
slots. Our `--stages 6 --batch-k 2` tests that depth; this does not claim the
upstream selector chooses this candidate. Our profile also tests shallower,
higher-residency queues. Stage 14 duplicates the rings and needs its own budget.

### The FP32 epilogue is implemented without BF16 conversion

DeepGEMM's FP32 branch uses float2 shared stores followed by TMA. Stages 11/14
preserve this arithmetic/output contract. Writers wait for previous async reads
before shared-buffer reuse, fence to the async proxy, synchronize only their
128-thread group, and issue TMA from an elected thread. Global store completion
is drained before exit. Named barriers keep independent producers/consumers
from accidentally joining a CTA-wide barrier in the steady loop.

At split=4 uniform split-K writes 4 MiB of partials; reduction reads those and
writes 1 MiB of C. Stage 12 changes this decomposition rather than merely
making the partial stores faster. It allocates two tile slots per worker,
which is not guaranteed smaller than uniform workspace. The last arriving
contributor reduces in fixed owner order; partial reads bypass stale L1 lines
across graph replays. Counters reset within each complete invocation. Workspace
is owned by a plan used sequentially on its captured stream.

### Multicast requires both an arm and a release protocol

At BN=64, summed A requests across output tiles are 64 MiB, despite only 1 MiB
of unique A. A two-CTA N cluster can reduce those input requests. It leaves B's
64 MiB unique data unchanged. Stage 13 arms both expected-byte barriers before
A multicast, then each consumer warp releases both replicas after safe MMA
completion. The producer drains the final slot generations and a final cluster
barrier protects shared storage lifetime. Cluster support and active-cluster
capacity are queried on the actual MIG device; unsupported cases are skipped.

### Ping-pong alternates whole output tiles

Stage 14 does not pad M to128 or assign consumers to duplicate rows. Consumer
WGs own alternating tile jobs; a shared math token passes only after the
current tile's WGMMA drain. The other WG can then compute while the former
runs its FP32 epilogue. One producer issuer probes both rings without blocking
on only one group. Full producer-WG register release is legal here; adding
`setmaxnreg` to the 32-thread producer in stages 6–13 would not be the same
collective. Reserving one CTA per SM protects the register budget but may hurt
memory latency hiding. This is a candidate to measure, not a promised winner.

## Remaining exclusions

The new paths implement the applicable mechanisms identified in the audit,
with specialized policies for one aligned shape. They do not reproduce the
libraries' complete candidate-scoring model, grouped/ragged scheduling,
autotuner, every cache policy or every epilogue. There is no claim of feature
parity or optimality. BN=16/256 and deeper slots are candidates, not generalized
edge handling. The FP32 contract and single M tile explain the BF16-output,
FP8 and Hilbert exclusions above.

[C-mma]: https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp#L474-L576
[C-prefetch]: https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp#L264-L276
[C-pingpong]: https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L127-L156
[C-streamk]: https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp#L379-L429
[D-desc]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L214-L222
[D-prod]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L106-L200
[D-loop]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L237-L281
[D-merge]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L45-L56
[D-prefetch]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L84-L89
[D-registers]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L126-L153
[D-epilogue]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L297-L386
[D-bf16-store]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh#L302-L347
[D-candidates]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/heuristics/sm90.hpp#L16-L108
[D-budget]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/heuristics/sm90.hpp#L148-L199
[D-scheduler]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L185-L264
[F-core]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_4.cuh#L269-L337
[F-registers]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_5.cuh#L270-L345
[F-persistent]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_6.cuh#L314-L389
[F-cluster]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_9.cuh#L383-L467
[F-zero]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_9.cuh#L475-L559
[F-3d]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_10.cuh#L32-L58
[F-epilogue]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_10.cuh#L544-L581
[F-hilbert]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_11.cuh#L535-L633
[F-stmatrix]: https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_12.cuh#L432-L535
[D-swizzle]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L108-L146
[D-host]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp#L115-L124
