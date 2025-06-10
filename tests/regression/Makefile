ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C basic
	$(MAKE) -C demo
	$(MAKE) -C dogfood
	$(MAKE) -C mstress
	$(MAKE) -C io_addr
	$(MAKE) -C printf
	$(MAKE) -C diverge
	$(MAKE) -C sort
	$(MAKE) -C fence
	$(MAKE) -C vecaddx
	$(MAKE) -C sgemmx
	$(MAKE) -C conv3x
	$(MAKE) -C sgemm2x
	$(MAKE) -C stencil3d

run-simx:
	$(MAKE) -C basic run-simx
	$(MAKE) -C demo run-simx
	$(MAKE) -C dogfood run-simx
	$(MAKE) -C mstress run-simx
	$(MAKE) -C io_addr run-simx
	$(MAKE) -C printf run-simx
	$(MAKE) -C diverge run-simx
	$(MAKE) -C sort run-simx
	$(MAKE) -C fence run-simx
	$(MAKE) -C vecaddx run-simx
	$(MAKE) -C sgemmx run-simx
	$(MAKE) -C conv3x run-simx
	$(MAKE) -C sgemm2x run-simx
	$(MAKE) -C stencil3d run-simx

run-rtlsim:
	$(MAKE) -C basic run-rtlsim
	$(MAKE) -C demo run-rtlsim
	$(MAKE) -C dogfood run-rtlsim
	$(MAKE) -C mstress run-rtlsim
	$(MAKE) -C io_addr run-rtlsim
	$(MAKE) -C printf run-rtlsim
	$(MAKE) -C diverge run-rtlsim
	$(MAKE) -C sort run-rtlsim
	$(MAKE) -C fence run-rtlsim
	$(MAKE) -C vecaddx run-rtlsim
	$(MAKE) -C sgemmx run-rtlsim
	$(MAKE) -C conv3x run-rtlsim
	$(MAKE) -C sgemm2x run-rtlsim
	$(MAKE) -C stencil3d run-rtlsim

run-opaesimx:
	$(MAKE) -C basic run-opaesimx
	$(MAKE) -C demo run-opaesimx
	$(MAKE) -C dogfood run-opaesimx
	$(MAKE) -C mstress run-opaesimx
	$(MAKE) -C io_addr run-opaesimx
	$(MAKE) -C printf run-opaesimx
	$(MAKE) -C diverge run-opaesimx
	$(MAKE) -C sort run-opaesimx
	$(MAKE) -C fence run-opaesimx
	$(MAKE) -C vecaddx run-opaesimx
	$(MAKE) -C sgemmx run-opaesimx
	$(MAKE) -C conv3x run-opaesimx
	$(MAKE) -C sgemm2x run-opaesimx
	$(MAKE) -C stencil3d run-opaesimx

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C demo clean
	$(MAKE) -C dogfood clean
	$(MAKE) -C mstress clean
	$(MAKE) -C io_addr clean
	$(MAKE) -C printf clean
	$(MAKE) -C diverge clean
	$(MAKE) -C sort clean
	$(MAKE) -C fence clean
	$(MAKE) -C vecaddx clean
	$(MAKE) -C sgemmx clean
	$(MAKE) -C conv3x clean
	$(MAKE) -C sgemm2x clean
	$(MAKE) -C stencil3d clean