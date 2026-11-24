NVCC ?= nvcc
OUT_DIR ?= ../build
# Avoid -arch=sm_90a shorthand: CUDA 12.6 also emits generic compute_90 PTX,
# which cannot support this tutorial's unconditionally emitted WGMMA code.
COMMON_FLAGS := -std=c++17 -O3 -DNDEBUG -lineinfo -I. -Iinclude
HOPPER_TARGET := -gencode=arch=compute_90a,code=sm_90a
LIBS := -lcublas -lcuda
HEADERS := $(wildcard include/*.cuh kernels/*.cuh)

.PHONY: all model help
all: $(OUT_DIR)/narrow_gemm

$(OUT_DIR):
	mkdir -p $@

$(OUT_DIR)/narrow_gemm: bench.cu $(HEADERS) Makefile | $(OUT_DIR)
	$(NVCC) $(COMMON_FLAGS) $(NVCCFLAGS) $(HOPPER_TARGET) $< -o $@ $(LIBS)

model:
	python3 scripts/model.py

help:
	@echo 'make [NVCC=/path/to/nvcc] [OUT_DIR=../build]'
	@echo 'All stages are handwritten CUDA/PTX; no CUTLASS/DeepGEMM checkout needed.'
	@echo 'Binaries, profiles, downloads and benchmark output belong outside this repo.'
