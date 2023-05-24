//
// Created by placek on 24.05.23.
//

#include <string>



#ifndef JSON_FILTERING_ALGORITHM_CUH
#define JSON_FILTERING_ALGORITHM_CUH

namespace filtering {
    template<int Index>
    size_t thread_filter(std::string &lines);
}

#endif //JSON_FILTERING_ALGORITHM_CUH
