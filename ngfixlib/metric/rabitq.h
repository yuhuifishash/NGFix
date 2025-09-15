#pragma once

#include "space.h"
#include "rabitqlib/defines.hpp"
#include <immintrin.h>

namespace ngfixlib{
    
class IPSpace_RaBitQ : public Space<uint8_t>
{
    size_t bits_b, bits_q;
public:
    float (IPSpace_RaBitQ::*fp) (const uint8_t*, const uint8_t*);
    virtual float dist_func(const uint8_t* vec0, const uint8_t* vec1) {
        return (this->*fp)(vec0, vec1);
    }

    // (delta_b * vec0[i] + vl_b) * (delta_q * vec1[i] + vl_q)
    // delta_b*delta_q * sum_{i=0}^d (vec0[i]*vec1[i])
    // delta_b*vl_q * sum_{i=0}^d (vec0[i])
    // delta_q*vl_b * sum_{i=0}^d (vec1[i])
    // d*vl_b*vl_q
    float dist_func_ip_b8_q8(const uint8_t* vec0, const uint8_t* vec1) {
        size_t d = dim - 8; // 0 <= i < d, d % 64 == 0
        __mmask32 mask = 0xFFFFFFFF;
        const uint32_t n = (d >> 5);
        __m512i sum512 = _mm512_set1_epi32(0);
        for (uint32_t i = 0; i < n; ++i) {
            __m256i v1 = _mm256_maskz_loadu_epi8(mask, vec0 + (i << 5));
            __m512i v1_512 = _mm512_cvtepu8_epi16(v1);
            __m256i v2 = _mm256_maskz_loadu_epi8(mask, vec1 + (i << 5));
            __m512i v2_512 = _mm512_cvtepu8_epi16(v2);
            sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
        }
        float res = _mm512_reduce_add_epi32(sum512);
        float* factor0 = (float*)(vec0 + d), *factor1 = (float*)(vec1 + d);
        float s1_b = factor0[1], s1_q = factor1[1];
        float delta_b = factor0[0], delta_q = factor1[0];
        // vl = - delta * (2^bit - 1) / 2]
        float vl_b = -delta_b * ((1 << bits_b) - 1) / 2, vl_q = -delta_q * ((1 << bits_q) - 1) / 2;
        return 1 - (res*delta_b*delta_q + delta_b*vl_q*s1_b + delta_q*vl_b*s1_q + d*vl_b*vl_q);
    }

    // query_data : raw_vector
    // (delta_b*vec0[i] + vl_b)*vec1[i]
    // delta_b*vec0[i]*vec1[i] + vl_b*vec1[i]
    float dist_func_ip_b8_q32(const uint8_t* vec0, const uint8_t* _vec1) {
        float* vec1 = (float*)_vec1;
        size_t d = dim - 8;
        __m512 sum = _mm512_setzero_ps();
        for (int i = 0; i < d; i += 16) {
            __m128i x_vec8 = _mm_loadu_si128((__m128i*)(vec0 + i));
            __m512i x_vec32 = _mm512_cvtepu8_epi32(x_vec8);
            __m512 x_float = _mm512_cvtepi32_ps(x_vec32);    
            __m512 y_float = _mm512_loadu_ps(vec1 + i);
            sum = _mm512_fmadd_ps(x_float, y_float, sum);
        }
        float res =  _mm512_reduce_add_ps(sum);

        float* factor0 = (float*)(vec0 + d);
        float s1_b = factor0[1], s1_q = vec1[d];
        float delta_b = factor0[0];
        float vl_b = -delta_b * ((1 << bits_b) - 1) / 2;
        return 1 - (delta_b * res + vl_b * s1_q);
    }

    IPSpace_RaBitQ(size_t _dimension_, size_t _bits_b, size_t _bits_q) : Space<uint8_t>(_dimension_) {
        this->bits_b = _bits_b;
        this->bits_q = _bits_q;
        if(bits_b == 8 && bits_q == 8) {
            fp = &IPSpace_RaBitQ::dist_func_ip_b8_q8;
        } else if(bits_b == 8 && bits_q == 32){
            fp = &IPSpace_RaBitQ::dist_func_ip_b8_q32;
        } else {
            throw std::runtime_error("Unsupported bits number.");
        }
    }
};


class L2Space_RaBitQ : public Space<uint8_t>
{
    size_t bits_b, bits_q;
public:
    float (L2Space_RaBitQ::*fp) (const uint8_t*, const uint8_t*);
    virtual float dist_func(const uint8_t* vec0, const uint8_t* vec1) {
        return (this->*fp)(vec0, vec1);
    }

    // (delta_b * vec0[i] + vl_b - delta_q * vec1[i] - vl_q)^2
    // (delta_b * vec0[i] - delta_q * vec1[i] + vl_b - vl_q)^2
    // ---------------------------------------------
    // 2*(delta_b * vec0[i] - delta_q * vec1[i])*(vl_b - vl_q) + (vl_b - vl_q)^2
    // +2*delta_b*(vl_b - vl_q)*sum_{i=0}^d (vec0[i]) 
    // -2*delta_q*(vl_b - vl_q)*sum_{i=0}^d (vec1[i]) 
    // +d*(vl_b - vl_q)^2
    // ---------------------------------------------
    // (delta_b * vec0[i] - delta_q * vec1[i])^2
    // +delta_b^2 * sum_{i=0}^d (vec0[i]^2)
    // +delta_q^2 * sum_{i=0}^d (vec1[i]^2)
    // -2*delta_b*delta_q * sum_{i=0}^d (vec0[i]*vec1[i])
    float dist_func_l2_b8_q8(const uint8_t* vec0, const uint8_t* vec1) {
        size_t d = dim - 12; // 0 <= i < d, d % 64 == 0
        __mmask32 mask = 0xFFFFFFFF;
        const uint32_t n = (d >> 5);
        __m512i sum512 = _mm512_set1_epi32(0);
        for (uint32_t i = 0; i < n; ++i) {
            __m256i v1 = _mm256_maskz_loadu_epi8(mask, vec0 + (i << 5));
            __m512i v1_512 = _mm512_cvtepu8_epi16(v1);
            __m256i v2 = _mm256_maskz_loadu_epi8(mask, vec1 + (i << 5));
            __m512i v2_512 = _mm512_cvtepu8_epi16(v2);
            sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
        }
        float res = _mm512_reduce_add_epi32(sum512);
        
        float* factor0 = (float*)(vec0 + d), *factor1 = (float*)(vec1 + d);
        float s2_b = factor0[2], s2_q = factor1[2];
        float s1_b = factor0[1], s1_q = factor1[1];
        float delta_b = factor0[0], delta_q = factor1[0];
        // vl = - delta * (2^bit - 1) / 2]
        float vl_b = -delta_b * ((1 << bits_b) - 1) / 2, vl_q = -delta_q * ((1 << bits_q) - 1) / 2;

        float res1 = -2*delta_b*delta_q*res;
        float res2 = delta_b*delta_b*s2_b + delta_q*delta_q*s2_q;
        float res3 = 2*delta_b*(vl_b - vl_q)*s1_b - 2*delta_q*(vl_b - vl_q)*s1_q + d*(vl_b - vl_q)*(vl_b - vl_q);
        return res1 + res2 + res3;
    }

    L2Space_RaBitQ(size_t _dimension_, size_t _bits_b, size_t _bits_q) : Space<uint8_t>(_dimension_) {
        this->bits_b = _bits_b;
        this->bits_q = _bits_q;
        if(bits_b == 8 && bits_q == 8) {
            fp = &L2Space_RaBitQ::dist_func_l2_b8_q8;
        } else {
            throw std::runtime_error("Unsupported bits number.");
        }
    }
};
}