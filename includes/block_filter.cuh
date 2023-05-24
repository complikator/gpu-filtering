//
// Created by placek on 30.05.23.
//

#ifndef JSON_FILTERING_BLOCK_FILTER_CUH
#define JSON_FILTERING_BLOCK_FILTER_CUH

#include <string>

namespace filtering {
    template<int Index>
    size_t block_filter(std::string &lines);
}

#endif //JSON_FILTERING_BLOCK_FILTER_CUH
