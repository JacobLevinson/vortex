#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE int32_t // output type
#endif

#define DOT8_INPUT_TYPE int8_t // A and B matrix element type

typedef struct {
  uint32_t grid_dim[2];
  uint32_t size;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif