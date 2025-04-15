#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto A = reinterpret_cast<uint32_t *>(arg->A_addr); // packed A
  auto B = reinterpret_cast<uint32_t *>(arg->B_addr); // packed B
  auto C = reinterpret_cast<TYPE *>(arg->C_addr);     // output
  auto size = arg->size;
  auto packed_width = (size + 3) / 4;

  int col = blockIdx.x;
  int row = blockIdx.y;

  int32_t sum = 0;
  for (int k = 0; k < packed_width; ++k) {
    uint32_t packedA = A[row * packed_width + k];
    uint32_t packedB = B[k * size + col]; // B is row-major
    sum += vx_dot8(packedA, packedB);
  }

  C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
