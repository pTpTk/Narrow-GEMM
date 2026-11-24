# Validation record

The authoring container has no NVIDIA GPU or system CUDA installation.
The CUDA compiler, headers, reference checkouts, object/PTX files, mock output
and logs live in the parent workspace, outside this repository.

## Checks completed for the handwritten implementation

- CUDA 12.6.85 compile-only validation for SM90a passed with all three BN
  variants of stages 2–7, plus the scalar kernel and FP32 reduction: 17 kernel
  instantiations in total. No warnings, WGMMA serialization diagnostics or
  register spills were reported. Flags included
  `-std=c++17 -O3 -DNDEBUG -lineinfo -arch=sm_90a -Xptxas=-v`.
- Compilation used only local source and CUDA/cuBLAS development include paths.
  CUTLASS, CuTe and DeepGEMM were absent from the build include path.
- An emitted-PTX audit found all 17 local kernel entries, no device function
  calls, and the expected `ldmatrix`, warp MMA, `cp.async`, TMA, WGMMA,
  mbarrier, and `setmaxnreg` instructions.
- The custom-kernel include audit found no CUTLASS, CuTe, DeepGEMM, WMMA,
  CUDA barrier/pipeline helper, or cuBLAS headers. The cuBLAS baseline is
  isolated in `include/reference.cuh`, included only by the benchmark driver.
- Python syntax, the arithmetic model, sweep CLI and CSV forwarding were
  checked. The sweep used an explicitly labeled mock executable: 65 processes,
  129 CSV records including local baselines, covering all supplied stage/BN/S
  sweep combinations. These checks do not execute GEMM.
- Makefile command generation, relative documentation paths and the absence
  of scratch artifacts from the deliverable were checked.

The current code has **not been linked against a complete installed cuBLAS
runtime or executed on an H100** here. Compilation and PTX inspection do not
establish numerical correctness, hardware descriptor acceptance, race freedom,
successful graph capture, dynamic-register behavior or performance.

## Compiler resource observations

These are ptxas reports for the current handwritten source, not GPU timings:

| Stage | BN | Registers / thread | Static shared bytes | Spill loads/stores |
| --- | ---: | ---: | ---: | ---: |
| 2 | 32 | 56 | 8,192 | 0 / 0 |
| 2 | 64 | 64 | 10,240 | 0 / 0 |
| 2 | 128 | 124 | 15,360 | 0 / 0 |
| 3 | 32 | 64 | 15,360 | 0 / 0 |
| 3 | 64 | 72 | 20,480 | 0 / 0 |
| 3 | 128 | 125 | 30,720 | 0 / 0 |
| 5 | 32 | 42 | dynamic | 0 / 0 |
| 5 | 64 | 58 | dynamic | 0 / 0 |
| 5 | 128 | 90 | dynamic | 0 / 0 |
| 6 | 32 | 42 | dynamic | 0 / 0 |
| 6 | 64 | 58 | dynamic | 0 / 0 |
| 6 | 128 | 90 | dynamic | 0 / 0 |
| 7 | 32 | 128 | dynamic | 0 / 0 |
| 7 | 64 | 128 | dynamic | 0 / 0 |
| 7 | 128 | 128 | dynamic | 0 / 0 |

Stages 5–7 use dynamic shared memory. Their launch setup prints the actual
allocation, including barriers and alignment padding. Stage 7 explicitly
requests a 128-register consumer budget and releases producer registers to 40;
its reported register count does not mean all warpgroups retain 128 registers
throughout execution. Compiler versions and template choices can change these
figures. No occupancy or throughput is inferred from this table alone.

## Checks to run on H100

1. Build with a complete CUDA 12.6 toolkit and cuBLAS. No CUTLASS/DeepGEMM
   checkout is needed.
2. Run every stage with random inputs. Sweep every BN and split value, including
   split 32 for the shortest K interval and persistent ring-boundary cases.
3. Repeat with `ones` and `alternating` inputs. Both timing modes should pass
   all-output comparisons before and after timing.
4. Run Compute Sanitizer memory and race checks, particularly on stages 6–7.
   The README gives commands. Inspect asynchronous operand lifetimes and
   mbarrier generations across persistent jobs.
5. Collect unprofiled complete-operation times. Profile partial GEMM and
   reduction separately to explain costs, while reporting their combined time.
6. If tuning the ring or register budgets, repeat correctness and resource
   checks. Compiler acceptance is insufficient evidence for a correct pipeline.

## Interpretation limits

- No performance table is measured. Roofline, traffic and grid numbers are arithmetic.
- cuBLAS is `GemmEx`, not a tuned cuBLASLt algorithm search; it is reference-only.
- Output and partials are FP32. This is not a BF16-output or FP8 benchmark.
- Timing is repeated-address steady state. No cold-cache guarantee is made.
- Graph construction and tensor-map encoding are untimed; reduction is timed.
- One aligned shape is supported. Editing dimensions requires revisiting
  bounds, fragment layouts, reduction sizes, TMA boxes, descriptor fields,
  shared alignment and workspace sizes.
