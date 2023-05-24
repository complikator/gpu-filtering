//
// Created by placek on 30.05.23.
//

#include "kmp.cuh"

__device__ void compute_lps_array(const char *pattern, size_t pattern_size, int *lps) {
    int len = 0;
    lps[0] = 0;

    int i = 1;
    while (i < pattern_size) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}
