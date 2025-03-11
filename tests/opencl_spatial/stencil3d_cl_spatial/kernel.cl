#include "common.h"

__kernel void stencil3d_cl(__global const TYPE *A,
                           __global TYPE *B,
                           int size)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int dep = get_global_id(2);

    int index = dep * size * size + row * size + col;

    TYPE sum = 0;
    int count = 0;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nz = clamp(dep + dz, 0, size - 1);
                int ny = clamp(row + dy, 0, size - 1);
                int nx = clamp(col + dx, 0, size - 1);
                sum += A[nz * size * size + ny * size + nx];
                count++;
            }
        }
    }
    B[index] = sum / count;
}
