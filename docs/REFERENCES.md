# Source guide

Sources were fetched and read on **2026-11-23**. These links pin code rather
than relying on a moving `main`. This repository contains original educational
CUDA C++ and inline PTX; it does not include or call CUTLASS, CuTe or DeepGEMM.
Upstream licenses still apply to their own repositories.

The expanded **2026-11-24** [source audit](SOURCE_AUDIT.md) also reads fast.cu
stages 4–12, DeepGEMM's host dispatch/MMA/TMA helpers and CUTLASS's pipelined
mainloop/ping-pong schedule. It records actual coverage; a reference link here
does not imply that every mechanism in that source is implemented locally.

## Worklog and organization

- Pranjal Shankhdhar, [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog),
  November 29, 2024. The inspiration is its numbered experiments, resource
  accounting, producer/consumer explanation, and measurement-led progression.
  Its square-GEMM speedups are not evidence for this tutorial's narrow shape.
- [fast.cu, pinned tree](https://github.com/pranjalssh/fast.cu/tree/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100):
  [driver](https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul.cu)
  and [early Tensor Core stage](https://github.com/pranjalssh/fast.cu/blob/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/h100/matmul/matmul_2.cuh).
  We borrow the small-driver/numbered-header organization, not its square tile
  parameters, correctness thresholds, or performance numbers.

## CUTLASS

The inspected source reference is **v3.9.2**, commit
[`ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`](https://github.com/NVIDIA/cutlass/tree/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e).

| Source | What to inspect | Connection to this tutorial |
| --- | --- | --- |
| [Hopper builder example](https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) | Tile, pipeline and epilogue resource choices | Background only; no builders are called |
| [SM90 TMA warp-specialized kernel](https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp) | Producer/consumer groups and shared-storage lifetimes | Input rings in 5–10; FP32 TMA epilogue in 11; two-consumer ping-pong in 14 |
| [SM90 pipeline](https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/pipeline/sm90_pipeline.hpp) | Full/empty transaction barriers and ring phases | Compare with our explicit barrier protocol and cross-job phase counters |
| [PTX barrier wrappers](https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/arch/barrier.h) | Initialization fences, parity waits and transaction byte counts | Instruction semantics checked against our local PTX wrappers |
| [TMA instruction wrappers](https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cute/arch/copy_sm90_tma.hpp) | Descriptor and shared address operands | Instruction reference; the helper is not included or called |
| [SM90 Stream-K scheduler](https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp) | Partitioning, partial reduction and fixup | Stage 12 implements contiguous tile/K partitions and last-arrival fixup; its policy/protocol is locally specialized |
| [Cooperative Stream-K tests](https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cooperative_stream_k.cu) | M tiles and cooperative schedules | Compare upstream M tiles/schedules with our M=64 implementation |

The pinned source checkout remains outside the deliverable repository. Its
headers are not on the build's include path. We study the implementation and
write the relevant specialized instruction/dataflow code locally.

## DeepGEMM

Source snapshot:
[`5081f8513af57e61696de4f4f04b0e707ffb503e`](https://github.com/deepseek-ai/DeepGEMM/tree/5081f8513af57e61696de4f4f04b0e707ffb503e).
This is a reference-only snapshot, not a dependency or a claimed benchmark
comparison. We inspect the **SM90 BF16 path**, keeping FP8 techniques separate.

| Source | Observed implementation | Lesson for narrow GEMM |
| --- | --- | --- |
| [sm90_bf16_gemm.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh) | Full/empty transaction barriers; producer TMA and consumer WGMMA; register deallocation/allocation; shared-memory output and TMA stores | Descriptor prefetch/updates in 9; FP32 shared/TMA output in 11; full-WG register redistribution in 14 |
| [SM90 heuristics](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/heuristics/sm90.hpp) | `get_layout_candidates`, `get_pipeline_config`, `get_launch_config`, `get_layout_info` | Jointly consider tile shape, available stages, epilogue footprint and wave utilization |
| [Persistent GEMM scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/gemm.cuh) | Swizzled block coordinates and repeated acquisition of logical blocks | Persistent workers and locality-aware ordering; neither alone creates missing logical work |

Specific observations from that snapshot:

- The heuristic's ordinary GEMM M candidates include 64 and 128; smaller M
  candidates are also added for very small M. We select 64 for exactly 64 rows.
- Its SM90 launch configuration uses 128 math threads for block M <= 64,
  and 256 for larger M; it also provisions a TMA producer group.
- Pipeline capacity subtracts output, barriers and other metadata before
  dividing available shared memory by per-stage storage.
- The candidate filter requires at least three stages, and at least four for
  its small tiles. This is evidence of joint tuning, not a requirement imposed
  by the TMA instruction.
- Its BF16 kernel waits for WGMMA completion before signaling an operand slot
  empty. More advanced overlap must preserve that operand lifetime.
- Some stages can be merged to reduce WGMMA-wait overhead under specific
  conditions. Our optional two-chunk groups test the same barrier/wait
  amortization idea with a bounded candidate set; stages 9–14 extend the legal queue depths to 2/3/4/5/6/8.

## Hardware and instruction documentation

- [H100 Tensor Core GPU datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-h100-tensor-core-gpu-datasheet):
  SKU-dependent peak throughput, bandwidth and capacity. Use dense BF16 figures
  when comparing with this dense GEMM, not sparse throughput.
- [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html):
  occupancy limits, TMA and thread-block clusters.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html):
  asynchronous copies, tensor-map encoding and synchronization.
- [PTX ISA, asynchronous warpgroup MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions):
  supported shapes, shared-memory descriptors, fences, commit and wait groups.
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html):
  `cublasGemmEx`, leading dimensions, compute modes and math flags.

The code claims are tied to the source files inspected above. Hardware and
instruction links are further reading; their moving documentation versions are
not a substitute for the pinned implementation and compile checks.
