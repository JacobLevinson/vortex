#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include "common.h"

#define FLOAT_ULP 6

using input_t = DOT8_INPUT_TYPE; // resolves to int8_t from common.h

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int32_t> {
public:
  static const char *type_str() { 
    return "int32_t"; 
  }
  static int32_t generate() {
    return rand(); 
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static int generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(TYPE *out, const uint32_t *A, const uint32_t *B, uint32_t width, uint32_t height) {

  uint32_t packed_width = (width + 3) / 4;

  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      int32_t sum = 0;
      for (uint32_t k = 0; k < packed_width; ++k) {
        uint32_t packedA = A[row * packed_width + k];
        uint32_t packedB = B[k * size + col]; // treats B as row-major
        sum += vx_dot8(packedA, packedB);
      }
      out[row * width + col] = sum;
    }
  }
}

  const char *kernel_file = "kernel.vxbin";
  uint32_t size = 32;

  vx_device_h device = nullptr;
  vx_buffer_h A_buffer = nullptr;
  vx_buffer_h B_buffer = nullptr;
  vx_buffer_h C_buffer = nullptr;
  vx_buffer_h krnl_buffer = nullptr;
  vx_buffer_h args_buffer = nullptr;
  kernel_arg_t kernel_arg = {};

  static void show_usage() {
    std::cout << "Vortex Test." << std::endl;
    std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
  }

  static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
      switch (c) {
      case 'n':
        size = atoi(optarg);
        break;
      case 'k':
        kernel_file = optarg;
        break;
      case 'h':
        show_usage();
        exit(0);
        break;
      default:
        show_usage();
        exit(-1);
      }
    }
  }

  void cleanup() {
    if (device) {
      vx_mem_free(A_buffer);
      vx_mem_free(B_buffer);
      vx_mem_free(C_buffer);
      vx_mem_free(krnl_buffer);
      vx_mem_free(args_buffer);
      vx_dev_close(device);
    }
  }

  int main(int argc, char *argv[]) {
    // parse command arguments
    parse_args(argc, argv);

    std::srand(50);

    // open device connection
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_dev_open(&device));

    uint32_t size_sq = size * size;
    uint32_t packed_width = (size + 3) / 4;
    uint32_t A_buf_size = size * packed_width * sizeof(uint32_t);
    uint32_t B_buf_size = size * packed_width * sizeof(uint32_t);
    uint32_t C_buf_size = size_sq * sizeof(TYPE);

    std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
    std::cout << "matrix size: " << size << "x" << size << std::endl;
    std::cout << "sizeof(packed A/B element)=" << sizeof(uint32_t) << ", sizeof(output C element)=" << sizeof(TYPE) << std::endl;

    kernel_arg.grid_dim[0] = size;
    kernel_arg.grid_dim[1] = size;
    kernel_arg.size = size;

    // allocate device memory
    std::cout << "allocate device memory" << std::endl;
    RT_CHECK(vx_mem_alloc(device, A_buf_size, VX_MEM_READ, &A_buffer));
    RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
    RT_CHECK(vx_mem_alloc(device, B_buf_size, VX_MEM_READ, &B_buffer));
    RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
    RT_CHECK(vx_mem_alloc(device, C_buf_size, VX_MEM_WRITE, &C_buffer));
    RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

    std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
    std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
    std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;

    // generate source data
    std::vector<uint32_t> h_A(size * packed_width);
    std::vector<uint32_t> h_B(size * packed_width);
    std::vector<TYPE> h_C(size_sq);
    for (uint32_t i = 0; i < size * packed_width; ++i) {
      h_A[i] = Comparator<TYPE>::generate(); // TYPE is int32_t
      h_B[i] = Comparator<TYPE>::generate();
    }

    // upload matrix A buffer
    {
      std::cout << "upload matrix A buffer" << std::endl;
      RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, A_buf_size));
    }

    // upload matrix B buffer
    {
      std::cout << "upload matrix B buffer" << std::endl;
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, B_buf_size));
    }

    // upload program
    std::cout << "upload program" << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    // upload kernel argument
    std::cout << "upload kernel argument" << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    auto time_start = std::chrono::high_resolution_clock::now();

    // start device
    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, C_buf_size));

    // verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
      std::vector<TYPE> h_ref(size_sq);
      matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size, size);

      for (uint32_t i = 0; i < h_ref.size(); ++i) {
        if (!Comparator<TYPE>::compare(h_C[i], h_ref[i], i, errors)) {
          ++errors;
        }
      }
    }

    // cleanup
    std::cout << "cleanup" << std::endl;
    cleanup();

    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return errors;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
  }