#pragma once

#include "space.h"
#include <immintrin.h>

namespace ngfixlib {

class L2Space_float : public Space<float>
{
public:
    //symqglib
    virtual float dist_func(const float* vec0, const float* vec1) {
        float result = 0;
        size_t mul16 = dim - (dim & 0b1111);
        auto sum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i < mul16; i += 16) {
            auto xxx = _mm512_loadu_ps(&vec0[i]);
            auto yyy = _mm512_loadu_ps(&vec1[i]);
            auto ttt = _mm512_sub_ps(xxx, yyy);
            sum = _mm512_fmadd_ps(ttt, ttt, sum);
        }
        result = _mm512_reduce_add_ps(sum);
        for (; i < dim; ++i) {
            float tmp = vec0[i] - vec1[i];
            result += tmp * tmp;
        }
        return result;
    };

    L2Space_float(size_t _dimension_) : Space<float>(_dimension_) {}
};

}