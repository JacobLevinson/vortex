###############################################################################
# runtime/opaesimx/Makefile – host-side Vortex runtime for the SimX OPAE shim
###############################################################################

ROOT_DIR := $(realpath ../..)
include ../common.mk

TARGET   ?= opaesimx            # allow TARGET= to pick Verilog vs SimX
DESTDIR  ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/runtime/opaesimx

# --------------------------------------------------------------------------- #
#  Compile flags
# --------------------------------------------------------------------------- #

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors -fPIC
CXXFLAGS += -I$(SRC_DIR) -I$(ROOT_DIR)/hw -I$(COMMON_DIR)
# add core-sim headers
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx         # contains arch.h, processor.h, …
CXXFLAGS += -I$(VORTEX_HOME)/sim/common
CXXFLAGS += -I$(VORTEX_HOME)/runtime/common          # util.h, dram_sim.h, …
CXXFLAGS += -I$(VORTEX_HOME)/runtime/include 
CXXFLAGS += -I$(ROOT_DIR)/runtime 
CXXFLAGS += -DXLEN_$(XLEN) $(CONFIGS)

# Choose debug or release
ifdef DEBUG
  CXXFLAGS += -g -O0
else
  CXXFLAGS += -O2 -DNDEBUG
endif

ifdef SCOPE
  CXXFLAGS += -DSCOPE
  SRCS += $(COMMON_DIR)/scope.cpp
endif

ifdef PERF
  CXXFLAGS += -DPERF_ENABLE
endif

# --------------------------------------------------------------------------- #
#  Source list
# --------------------------------------------------------------------------- #

SRCS = $(SRC_DIR)/vortex.cpp $(SRC_DIR)/driver.cpp

# Build dependency: shared simulator library
BUILD_DEPS = $(DESTDIR)/libopae-c-simx.so
CXXFLAGS  += -DOPAESIMX -I$(SIM_DIR)/opaesimx

# --------------------------------------------------------------------------- #
#  Link flags (same as original runtime/opae)
# --------------------------------------------------------------------------- #
LDFLAGS += -shared -ldl -pthread
# nothing else is required; libopae-c-simx.so is pulled in with -L/-l below
LDFLAGS += -L$(DESTDIR) -lopae-c-simx    # add rpath if desired

# --------------------------------------------------------------------------- #
#  Targets
# --------------------------------------------------------------------------- #
PROJECT := libvortex-opaesimx.so        # analogous to libvortex-opae.so

all: $(DESTDIR)/$(PROJECT)

# Rule that builds the dependency shared library in sim/opaesimx
$(DESTDIR)/libopae-c-simx.so:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/opaesimx $@

# Final runtime shared object
$(DESTDIR)/$(PROJECT): $(SRCS) $(BUILD_DEPS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean-driver:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/opaesimx clean

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime

.PHONY: all clean clean-driver clean-runtime
