ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C vecadd_spatial
	$(MAKE) -C sgemm_spatial
	$(MAKE) -C conv3_spatial
	$(MAKE) -C psort_spatial
	$(MAKE) -C saxpy_spatial
	$(MAKE) -C sfilter_spatial
	$(MAKE) -C sgemm2_spatial
	$(MAKE) -C sgemm3_spatial
	$(MAKE) -C psum_spatial
	$(MAKE) -C oclprintf_spatial
	$(MAKE) -C dotproduct_spatial
	$(MAKE) -C transpose_spatial
	$(MAKE) -C spmv_spatial
	$(MAKE) -C stencil3d_cl_spatial
	$(MAKE) -C stencil_spatial
	$(MAKE) -C lbm_spatial
	$(MAKE) -C nearn_spatial
	$(MAKE) -C guassian_spatial
	$(MAKE) -C kmeans_spatial
	$(MAKE) -C blackscholes_spatial
	$(MAKE) -C bfs_spatial

run-simx:
	$(MAKE) -C vecadd_spatial run-simx
	$(MAKE) -C sgemm_spatial run-simx
	$(MAKE) -C conv3_spatial run-simx
	$(MAKE) -C psort_spatial run-simx
	$(MAKE) -C saxpy_spatial run-simx
	$(MAKE) -C sfilter_spatial run-simx
	$(MAKE) -C sgemm2_spatial run-simx
	$(MAKE) -C sgemm3_spatial run-simx
	$(MAKE) -C psum_spatial run-simx
	$(MAKE) -C oclprintf_spatial run-simx
	$(MAKE) -C dotproduct_spatial run-simx
	$(MAKE) -C transpose_spatial run-simx
	$(MAKE) -C spmv_spatial run-simx
	$(MAKE) -C stencil3d_cl_spatial run-simx
	$(MAKE) -C stencil_spatial run-simx
	$(MAKE) -C nearn_spatial run-simx
	$(MAKE) -C guassian_spatial run-simx
	$(MAKE) -C kmeans_spatial run-simx
	$(MAKE) -C blackscholes_spatial run-simx
	$(MAKE) -C bfs_spatial run-simx

run-rtlsim:
	$(MAKE) -C vecadd_spatial run-rtlsim
	$(MAKE) -C sgemm_spatial run-rtlsim
	$(MAKE) -C conv3_spatial run-rtlsim
	$(MAKE) -C psort_spatial run-rtlsim
	$(MAKE) -C saxpy_spatial run-rtlsim
	$(MAKE) -C sfilter_spatial run-rtlsim
	$(MAKE) -C sgemm2_spatial run-rtlsim
	$(MAKE) -C sgemm3_spatial run-rtlsim
	$(MAKE) -C psum_spatial run-rtlsim
	$(MAKE) -C oclprintf_spatial run-rtlsim
	$(MAKE) -C dotproduct_spatial run-rtlsim
	$(MAKE) -C transpose_spatial run-rtlsim
	$(MAKE) -C spmv_spatial run-rtlsim
	$(MAKE) -C stencil3d_cl_spatial run-rtlsim
	$(MAKE) -C stencil_spatial run-rtlsim
	$(MAKE) -C nearn_spatial run-rtlsim
	$(MAKE) -C guassian_spatial run-rtlsim
	$(MAKE) -C kmeans_spatial run-rtlsim
	$(MAKE) -C blackscholes_spatial run-rtlsim
	$(MAKE) -C bfs_spatial run-rtlsim

clean:
	$(MAKE) -C vecadd_spatial clean
	$(MAKE) -C sgemm_spatial clean
	$(MAKE) -C conv3_spatial clean
	$(MAKE) -C psort_spatial clean
	$(MAKE) -C saxpy_spatial clean
	$(MAKE) -C sfilter_spatial clean
	$(MAKE) -C sgemm2_spatial clean
	$(MAKE) -C sgemm3_spatial clean
	$(MAKE) -C psum_spatial clean
	$(MAKE) -C oclprintf_spatial clean
	$(MAKE) -C dotproduct_spatial clean
	$(MAKE) -C transpose_spatial clean
	$(MAKE) -C spmv_spatial clean
	$(MAKE) -C stencil3d_cl_spatial clean
	$(MAKE) -C stencil_spatial clean
	$(MAKE) -C lbm_spatial clean
	$(MAKE) -C nearn_spatial clean
	$(MAKE) -C guassian_spatial clean
	$(MAKE) -C kmeans_spatial clean
	$(MAKE) -C blackscholes_spatial clean
	$(MAKE) -C bfs_spatial clean