#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include "common.h"

#define KERNEL_NAME "stencil3d_cl"
#define FLOAT_ULP 6

#define CL_CHECK(_expr)                                                 \
    do                                                                  \
    {                                                                   \
        cl_int _err = _expr;                                            \
        if (_err == CL_SUCCESS)                                         \
            break;                                                      \
        printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
        cleanup();                                                      \
        exit(-1);                                                       \
    } while (0)

#define CL_CHECK2(_expr)                                                    \
    ({                                                                      \
        cl_int _err = CL_INVALID_VALUE;                                     \
        decltype(_expr) _ret = _expr;                                       \
        if (_err != CL_SUCCESS)                                             \
        {                                                                   \
            printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
            cleanup();                                                      \
            exit(-1);                                                       \
        }                                                                   \
        _ret;                                                               \
    })

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
    static const char* type_str() { return "float"; }
    static float generate() { return static_cast<float>(rand()) / RAND_MAX; }
    static bool compare(float a, float b, int index, int errors) {
        union fi_t { float f; int32_t i; };
        fi_t fa, fb;
        fa.f = a; fb.f = b;
        auto d = std::abs(fa.i - fb.i);
        if (d > FLOAT_ULP) {
            if (errors < 100)
                printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
            return false;
        }
        return true;
    }
};

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
    if (nullptr == filename || nullptr == data || 0 == size)
        return -1;
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    *data = (uint8_t*)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);
    fclose(fp);
    return 0;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (a_memobj) clReleaseMemObject(a_memobj);
    if (b_memobj) clReleaseMemObject(b_memobj);
    if (context) clReleaseContext(context);
    if (device_id) clReleaseDevice(device_id);
    if (kernel_bin) free(kernel_bin);
}

// CPU Reference Implementation (3D Stencil)
static void stencil_cpu(TYPE *B, const TYPE *A, int size) {
    for (int z = 0; z < size; z++) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                TYPE sum = 0;
                int count = 0;
                for (int dz = -1; dz <= 1; dz++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int nz = std::max(0, std::min(z + dz, size - 1));
                            int ny = std::max(0, std::min(y + dy, size - 1));
                            int nx = std::max(0, std::min(x + dx, size - 1));
                            sum += A[nz * size * size + ny * size + nx];
                            count++;
                        }
                    }
                }
                B[z * size * size + y * size + x] = sum / count;
            }
        }
    }
}

uint32_t size = 32;
uint32_t block_size = 2;

static void show_usage() {
    printf("Usage: [-n size] [-b block_size] [-h help]\n");
}

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "n:b:h?")) != -1) {
        switch (c) {
            case 'n': size = atoi(optarg); break;
            case 'b': block_size = atoi(optarg); break;
            case 'h':
            case '?': show_usage(); exit(0);
            default: show_usage(); exit(-1);
        }
    }
    printf("Parsed Arguments: size=%d, block_size=%d\n", size, block_size);
}

int main (int argc, char **argv) {
    parse_args(argc, argv);
    size_t size_cubed = size * size * size;
    size_t nbytes = size_cubed * sizeof(TYPE);

    cl_platform_id platform_id;
    size_t kernel_size;

    srand(50);

    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

    printf("Create context\n");
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

    a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
    b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

    printf("Create program from kernel source\n");
    if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size)) {
        cleanup();
        return -1;
    }

    program = CL_CHECK2(clCreateProgramWithSource(context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
    kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_memobj));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &size));

    std::vector<TYPE> h_a(size_cubed), h_b(size_cubed);
    for (size_t i = 0; i < size_cubed; ++i) h_a[i] = Comparator<TYPE>::generate();

    commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

    printf("Upload source buffers\n");
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a.data(), 0, NULL, NULL));

    size_t global_work_size[3] = {size, size, size};
    size_t local_work_size[3] = {block_size, block_size, block_size};

    printf("Execute the kernel\n");
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

    CL_CHECK(clEnqueueReadBuffer(commandQueue, b_memobj, CL_TRUE, 0, nbytes, h_b.data(), 0, NULL, NULL));

    printf("Verify result\n");
    std::vector<TYPE> h_ref(size_cubed);
    stencil_cpu(h_ref.data(), h_a.data(), size);
    int errors = 0;
    for (size_t i = 0; i < size_cubed; ++i)
        if (!Comparator<TYPE>::compare(h_b[i], h_ref[i], i, errors)){
            ++errors;
        }

    if (errors != 0)
    {
        printf("FAILED! - %d errors\n", errors);
    }
    else
    {
        printf("PASSED!\n");
    }

    cleanup();
    return errors;
}
