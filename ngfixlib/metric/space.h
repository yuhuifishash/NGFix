#pragma once

namespace ngfixlib {

template<typename T>
class Space
{
public:
    size_t dim;
    virtual float dist_func(const T* vec0, const T* vec1) = 0;
    Space(size_t _dimension_) : dim(_dimension_) {}
};

}