ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := stencil3d_cl_spatial
SRC_DIR := $(VORTEX_HOME)/tests/opencl_spatial/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

common.h: $(SRC_DIR)/common.h
	cp $< $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl common.h

# Default test options (Matrix size: 32x32x32, Block size: 2x2x2)
OPTS ?= -n32-b2

include ../common.mk
