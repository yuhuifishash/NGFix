#pragma once

#include "space.h"
#include <immintrin.h>

namespace ngfixlib {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

class IPSpace_float : public Space<float>
{
public:
    float (IPSpace_float::*fp) (const float*, const float*);
    virtual float dist_func(const float* vec0, const float* vec1) {
        return (this->*fp)(vec0, vec1);
    };

    // hnswlib
    float ipDistanceSIMD16ExtAVX512(const float *pVect1v, const float *pVect2v) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = dim;
        size_t qty16 = qty / 16;

        const float *pEnd1 = pVect1 + 16 * qty16;

        __m512 sum512 = _mm512_set1_ps(0);

        size_t loop = qty16 / 4;
        
        while (loop--) {
            __m512 v1 = _mm512_loadu_ps(pVect1);
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            __m512 v3 = _mm512_loadu_ps(pVect1);
            __m512 v4 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            __m512 v5 = _mm512_loadu_ps(pVect1);
            __m512 v6 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            __m512 v7 = _mm512_loadu_ps(pVect1);
            __m512 v8 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            sum512 = _mm512_fmadd_ps(v1, v2, sum512);
            sum512 = _mm512_fmadd_ps(v3, v4, sum512);
            sum512 = _mm512_fmadd_ps(v5, v6, sum512);
            sum512 = _mm512_fmadd_ps(v7, v8, sum512);
        }

        while (pVect1 < pEnd1) {
            __m512 v1 = _mm512_loadu_ps(pVect1);
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;
            sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        }

        float sum = _mm512_reduce_add_ps(sum512);
        return 1 - sum;
    }


    float ipDistanceSIMD4ExtAVX512(const float *pVect1v, const float *pVect2v) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = dim;
        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m512 sum512 = _mm512_set1_ps(0);

        size_t loop = qty16 / 4;
        
        while (loop--) {
            __m512 v1 = _mm512_loadu_ps(pVect1);
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            __m512 v3 = _mm512_loadu_ps(pVect1);
            __m512 v4 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            __m512 v5 = _mm512_loadu_ps(pVect1);
            __m512 v6 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            __m512 v7 = _mm512_loadu_ps(pVect1);
            __m512 v8 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;

            sum512 = _mm512_fmadd_ps(v1, v2, sum512);
            sum512 = _mm512_fmadd_ps(v3, v4, sum512);
            sum512 = _mm512_fmadd_ps(v5, v6, sum512);
            sum512 = _mm512_fmadd_ps(v7, v8, sum512);
        }

        while (pVect1 < pEnd1) {
            __m512 v1 = _mm512_loadu_ps(pVect1);
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect1 += 16;
            pVect2 += 16;
            sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        }

        float sum = _mm512_reduce_add_ps(sum512);

        __m128 sum128 = _mm_set1_ps(0);
        while (pVect1 < pEnd2) {
            __m128 v1 = _mm_loadu_ps(pVect1);
            __m128 v2 = _mm_loadu_ps(pVect2);
            pVect1 += 4;
            pVect2 += 4;
            sum128 = _mm_fmadd_ps(v1, v2, sum128);
        }
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum += _mm_cvtss_f32(sum128);  
        
        return 1 - sum;
    }

    IPSpace_float(size_t _dimension_) : Space<float>(_dimension_) {
        if(dim%16 == 0) {
            fp = &IPSpace_float::ipDistanceSIMD16ExtAVX512;
        } else if(dim%4 == 0) {
            fp = &IPSpace_float::ipDistanceSIMD4ExtAVX512;
        } else {
            throw std::runtime_error("dim % 4 shoule be 0.");
        }
    }
};
}