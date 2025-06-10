include ../common.mk
DESTDIR ?= $(CURDIR)

SRC_DIR := $(VORTEX_HOME)/sim/opaesimx
SRCROOT := $(ROOT_DIR)/..

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Wno-array-bounds -fPIC
CXXFLAGS += -I$(SRCROOT)/sim/simx
CXXFLAGS += -I$(SRCROOT)/sim/common
CXXFLAGS += -I$(SRCROOT)/runtime/include
CXXFLAGS += -I$(SRCROOT)/runtime/common
CXXFLAGS += -I$(ROOT_DIR)/runtime
CXXFLAGS += -I$(SRC_DIR) -I$(ROOT_DIR)/hw -I$(COMMON_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/spdlog/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/yaml-cpp/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/src
CXXFLAGS += -DXLEN_$(XLEN) $(CONFIGS)

LDFLAGS  += -shared \
            $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a \
            -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator \
            -L$(THIRD_PARTY_DIR)/ramulator -lramulator \
	    -L$(DESTDIR) -lsimx \
            -ldl -pthread

BUILD_DEPS += $(DESTDIR)/libsimx.so


SRCS = \
    $(COMMON_DIR)/util.cpp \
    $(COMMON_DIR)/mem.cpp \
    $(COMMON_DIR)/softfloat_ext.cpp \
    $(COMMON_DIR)/rvfloats.cpp \
    $(COMMON_DIR)/dram_sim.cpp \
    $(SRC_DIR)/fpga.cpp \
    $(SRC_DIR)/opae_simx.cpp

PROJECT := libopae-c-simx.so

all: $(DESTDIR)/$(PROJECT)

$(DESTDIR)/libsimx.so:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/simx $@

$(DESTDIR)/$(PROJECT): $(SRCS)
	@mkdir -p $(DESTDIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC $^ $(LDFLAGS) -o $@

clean:
	rm -f $(DESTDIR)/$(PROJECT) *.o */*.o
