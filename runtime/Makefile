ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: simx opae opaesimx

vm: stub simx

stub:
	$(MAKE) -C stub

stubarm:
	$(MAKE) -C stubarm

simx:
	$(MAKE) -C simx

opae:
	$(MAKE) -C opae

opaesimx:
	$(MAKE) -C opaesimx

xrt:
	$(MAKE) -C xrt
	$(MAKE) -C simx clean
	$(MAKE) -C rtlsim clean
	$(MAKE) -C opae clean
	$(MAKE) -C opaesimx clean
	$(MAKE) -C xrt clean

.PHONY: all stub simx rtlsim opae opaesimx xrt clean
