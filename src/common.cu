//
// Created by placek on 30.05.23.
//

#include <common.cuh>

namespace filtering {
    __global__ void
    find_newlines_kernel(const char *jsons_file, const size_t jsons_file_size, bool *found_jsons_lines) {
        auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid >= jsons_file_size) {
            return;
        }

        if (jsons_file[tid] == JSON_SEPARATOR) {
            found_jsons_lines[tid] = true;
        }
    }
}