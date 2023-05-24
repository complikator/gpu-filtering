//
// Created by placek on 23.05.23.
//

#ifndef JSON_FILTERING_KMP_CUH
#define JSON_FILTERING_KMP_CUH

#include <string>
#include "utils.cuh"

extern __device__ void compute_lps_array(const char *pattern, size_t pattern_size, int *lps);

template<int Index>
__device__ bool is_pattern_present_in_text(const char *text, size_t text_size) {
    constexpr auto pattern_size = get_len<Index>();
    constexpr auto pattern = get_word<Index>();
    int lps[pattern_size];

    compute_lps_array(pattern, pattern_size, lps);

    int i = 0;
    int j = 0;
    while ((text_size - i) >= (pattern_size - j)) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }

        if (j == pattern_size) {
            return true;
        } else if (i < text_size && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }
    }

    return false;
}

#endif //JSON_FILTERING_KMP_CUH
