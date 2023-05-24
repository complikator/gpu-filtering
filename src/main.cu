#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>


#include <thread_filter.cuh>
#include <block_filter.cuh>
#include "config.h"
#include "utils.cuh"

int main() {
    // 1. Read the file
    std::ifstream input("test.json");
    std::stringstream ss;
    ss << input.rdbuf();
    // 2. Filter
    std::string res = ss.str();

    CUDA_MEASURE_TIME_START()

    std::cout << "block: " << filtering::block_filter<configuration::Index>(res) << std::endl;

    CUDA_MEASURE_TIME_END("block filter");

    CUDA_MEASURE_TIME_START()

    std::cout << "thread: " << filtering::thread_filter<configuration::Index>(res) << std::endl;

    CUDA_MEASURE_TIME_END("thread filter");
}