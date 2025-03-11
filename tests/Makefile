ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: unittest kernel regression opencl opencl_spatial riscv

unittest:
	$(MAKE) -C unittest

kernel:
	$(MAKE) -C kernel

regression:
	$(MAKE) -C regression

opencl:
ifneq ($(XLEN),64)
	$(MAKE) -C opencl
endif

opencl_spatial:
ifneq ($(XLEN),64)
	$(MAKE) -C opencl_spatial
endif

riscv:
	$(MAKE) -C riscv

clean:
	$(MAKE) -C unittest clean
	$(MAKE) -C kernel clean
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean
	$(MAKE) -C opencl_spatial clean
	$(MAKE) -C riscv clean

.PHONY: all unittest kernel regression opencl opencl_spatial riscv