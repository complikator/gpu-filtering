//
// Created by placek on 30.05.23.
//

#ifndef JSON_FILTERING_COMMON_CUH
#define JSON_FILTERING_COMMON_CUH

namespace filtering {
    constexpr char JSON_SEPARATOR = '\n';

    __global__ void
    find_newlines_kernel(const char *jsons_file, const size_t jsons_file_size, bool *found_jsons_lines);
}

#endif //JSON_FILTERING_COMMON_CUH
