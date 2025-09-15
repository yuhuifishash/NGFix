#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>

#include "index/lut.hpp"
#include "quantization/rabitq.hpp"
#include "utils/space.hpp"

namespace rabitqlib {
/**
 * @brief use an object to store data used for searching on symphonyqg for a given query
 */
template <typename T>
class BatchQuery {
   private:
    Lut<T> lookup_table_;
    T G_add_ = 0;
    T G_k1xSumq_ = 0;  // G_k1xSumq

   public:
    explicit BatchQuery(const T* rotated_query, size_t padded_dim) {
        lookup_table_ = std::move(Lut<T>(rotated_query, padded_dim));

        float c_1 = -((1 << 1) - 1) / 2.F;

        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
    }

    [[nodiscard]] T delta() const { return lookup_table_.delta(); }

    [[nodiscard]] T sum_vl_lut() const { return lookup_table_.sum_vl(); }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    void set_g_add(T sqr_norm) {
        G_add_ = sqr_norm;
    }  // may need to be edited if we want to support ip

    [[nodiscard]] const uint8_t* lut() const { return lookup_table_.lut(); }
};

template <typename T>
class SplitBatchQuery {
   private:
    const T* rotated_query_;
    Lut<T> lookup_table_;
    T G_add_ = 0;
    T G_error_ = 0;
    T G_k1xSumq_ = 0;
    T G_kbxSumq_ = 0;
    MetricType metric_type_ = METRIC_L2;

   public:
    explicit SplitBatchQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        MetricType metric_type = METRIC_L2,
        bool use_hacc = true
    )
        : rotated_query_(rotated_query) {
        lookup_table_ = std::move(Lut<T>(rotated_query, padded_dim, use_hacc));

        metric_type_ = (metric_type == METRIC_IP) ? METRIC_IP : METRIC_L2;

        float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
        float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
        G_kbxSumq_ = sumq * c_b;
    }
    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T delta() const { return lookup_table_.delta(); }

    [[nodiscard]] T sum_vl_lut() const { return lookup_table_.sum_vl(); }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    void set_g_add(T norm, T ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = norm * norm;
            G_error_ = norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -ip;
            G_error_ = norm;
        }
    }

    [[nodiscard]] const uint8_t* lut() const { return lookup_table_.lut(); }
};

template <typename T>
class SplitSingleQuery {
   private:
    const T* rotated_query_;
    std::vector<uint64_t> QueryBin_;
    T G_add_;
    T G_k1xSumq_;
    T G_kbxSumq_;
    T G_error_;
    T delta_;
    T vl_;
    MetricType metric_type_ = METRIC_L2;

   public:
    static constexpr size_t kNumBits = 4;
    explicit SplitSingleQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        quant::RabitqConfig config,
        size_t metric_type = METRIC_L2
    )
        : rotated_query_(rotated_query), QueryBin_(padded_dim * kNumBits / 64, 0) {
        float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
        float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
        G_kbxSumq_ = sumq * c_b;

        metric_type_ = (metric_type == METRIC_IP) ? METRIC_IP : METRIC_L2;

        std::vector<uint16_t> quant_query = std::vector<uint16_t>(padded_dim);

        // quantize query by rabitq
        quant::quantize_scalar<float, uint16_t>(
            rotated_query, padded_dim, kNumBits, quant_query.data(), delta_, vl_, config
        );

        // represent quantized query as u64
        rabitqlib::new_transpose_bin(
            quant_query.data(), QueryBin_.data(), padded_dim, kNumBits
        );
    }

    [[nodiscard]] const uint64_t* query_bin() const { return QueryBin_.data(); }

    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T delta() const { return delta_; }

    [[nodiscard]] T vl() const { return vl_; }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    void set_g_add(T norm, T ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = norm * norm;
            G_error_ = norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -ip;
            G_error_ = norm;
        }
    }

    void set_g_error(T norm) { G_error_ = norm; }
};

}  // namespace rabitqlib