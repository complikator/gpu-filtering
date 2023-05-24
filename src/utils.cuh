//
// Created by placek on 25.05.23.
//

#ifndef JSON_FILTERING_UTILS_CUH
#define JSON_FILTERING_UTILS_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iomanip>

#define CUDA_CHECK(x) do {\
        cudaError_t err = x;\
        if (err != cudaSuccess) {\
            const char* errname = cudaGetErrorName(err);\
            const char* errdesc = cudaGetErrorString(err);\
            printf("ERROR, file: %s, line: %d: Cuda call failed: %s: %s \n", __FILE__, __LINE__, errname, errdesc);\
            exit(-1);\
        }\
    }\
    while(0)

#define TID (blockIdx.x * blockDim.x + threadIdx.x)

#define CUDA_KERNEL_FINISH()  \
CUDA_CHECK(cudaDeviceSynchronize()); \
CUDA_CHECK(cudaGetLastError())


__device__ const char d_string1[] = "first";

template <int Index> __device__ constexpr const char *get_word() {
    return d_string1;
}
template <int Index> __device__ constexpr size_t get_len() { return 5; }

#define CUDA_MEASURE_TIME_START()  do {\
cudaEvent_t start, stop; \
CUDA_CHECK(cudaEventCreate(&start)); \
CUDA_CHECK(cudaEventCreate(&stop)); \
CUDA_CHECK(cudaEventRecord(start)); \

#define CUDA_MEASURE_TIME_END(name)\
CUDA_CHECK(cudaEventRecord(stop));\
CUDA_CHECK(cudaEventSynchronize(stop));\
float time; \
CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));\
std::cout  << std::left << "procedure <" << std::setw(20) << name << "> took: " << std::setw(10)<< time << " ms" << std::endl;  \
} while (0)
#endif //JSON_FILTERING_UTILS_CUH
