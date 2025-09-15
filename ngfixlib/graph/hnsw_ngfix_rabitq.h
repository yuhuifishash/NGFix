#pragma once

#include "hnsw_ngfix.h"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace ngfixlib {
class HNSW_NGFix_RaBitQ {
public:
    size_t dim;
    size_t padded_dim;
    size_t b_bits;
    size_t q_bits;
    rabitqlib::quant::RabitqConfig config;
    Metric metric;

    rabitqlib::Rotator<float>* rotator;
    HNSW_NGFix<uint8_t>* hnsw_ngfix;

    // rerank
    float* rawdata = nullptr;
    bool is_rerank = false;
    float rerank_ratio = 2;
    Space<float>* rerank_space = nullptr;

    HNSW_NGFix_RaBitQ(Metric metric, size_t _dimension, size_t max_elements, 
                        size_t _b_bits = 8, size_t M_ = 16, size_t MEX_ = 48)  {
        this->metric = metric;
        this->dim = _dimension;
        this->rotator = rabitqlib::choose_rotator<float>(_dimension);
        this->padded_dim = rotator->size(); // padded_dim
        this->b_bits = _b_bits;
        this->config = rabitqlib::quant::faster_config(padded_dim, b_bits);
        // query and base use the same number of bits by default
        this->q_bits = b_bits;
        if (metric == L2_RaBitQ && b_bits == 8 && q_bits == 8) {
            hnsw_ngfix = new HNSW_NGFix<uint8_t>(L2_RaBitQ_b8_q8, padded_dim + 12, max_elements, M_, MEX_);
        } else if(metric == IP_RaBitQ && b_bits == 8 && q_bits == 8) {
            hnsw_ngfix = new HNSW_NGFix<uint8_t>(IP_RaBitQ_b8_q8, padded_dim + 8, max_elements, M_, MEX_);
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }
        
    }

    HNSW_NGFix_RaBitQ(Metric metric, std::string path) {
        std::ifstream input(path + ".meta", std::ios::binary);
        input.read((char*)&b_bits, sizeof(b_bits));
        input.read((char*)&q_bits, sizeof(q_bits));
        input.read((char*)&padded_dim, sizeof(padded_dim));
        input.read((char*)&dim, sizeof(dim));
        
        this->rotator = rabitqlib::choose_rotator<float>(dim);
        this->rotator->load(input);

        this->metric = metric;
        this->config = rabitqlib::quant::faster_config(padded_dim, b_bits);
        // query and base use the same number of bits by default
        if (metric == L2_RaBitQ && b_bits == 8 && q_bits == 8) {
            hnsw_ngfix = new HNSW_NGFix<uint8_t>(L2_RaBitQ_b8_q8, path);
        } else if(metric == IP_RaBitQ && b_bits == 8 && q_bits == 8) {
            hnsw_ngfix = new HNSW_NGFix<uint8_t>(IP_RaBitQ_b8_q8, path);
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }
    }

    void StoreIndex(std::string path) {
        std::ofstream output(path + ".meta", std::ios::binary);
        output.write((char*)&b_bits, sizeof(b_bits));
        output.write((char*)&q_bits, sizeof(q_bits));
        output.write((char*)&padded_dim, sizeof(padded_dim));
        output.write((char*)&dim, sizeof(dim));
        rotator->save(output);
        hnsw_ngfix->StoreIndex(path);
    }

    ~HNSW_NGFix_RaBitQ() {
        delete hnsw_ngfix;
        delete rotator;
        if(rerank_space) {
            delete rerank_space;
        }
    }

    void rerank_init(float* _raw_data) {
        this->is_rerank = true;
        this->rawdata = _raw_data;
        if(rerank_space) {
            delete rerank_space;
        }
        if(metric == IP_RaBitQ) {
            rerank_space = new IPSpace_float(dim);
        } else if(metric == L2_RaBitQ) {
            rerank_space = new L2Space_float(dim);
        }

        if(b_bits == 8 && q_bits == 8) {
            rerank_ratio = 1.5;
        } else if(b_bits == 8 && q_bits == 32) {
            rerank_ratio = 1.5;
        } else {
            throw std::runtime_error("Unknown bits.");
        }
    }

    void set_q_bits(size_t bits) {
        this->q_bits = bits;
        if (metric == L2_RaBitQ && b_bits == 8 && q_bits == 8) {
            hnsw_ngfix->set_new_metric(L2_RaBitQ_b8_q8);
        } else if(metric == IP_RaBitQ && b_bits == 8 && q_bits == 8) {
            hnsw_ngfix->set_new_metric(IP_RaBitQ_b8_q8);
        } else if(metric == IP_RaBitQ && b_bits == 8 && q_bits == 32) {
            hnsw_ngfix->set_new_metric(IP_RaBitQ_b8_q32);
        } else {
            throw std::runtime_error("Error: Unsupported bits number");
        }
    }

    void getRaBitQCode_ip(float* data, uint8_t* code, size_t bits) {
        if(bits == 32) { // raw data
            rotator->rotate(data, (float*)code);
            float* f_code = (float*)code;
            f_code[padded_dim] = std::accumulate(f_code, f_code + padded_dim, 0.0f);
            return;
        }   

        float* rotated_data = new float[padded_dim];
        rotator->rotate(data, rotated_data);
        float delta, vl;
        rabitqlib::quant::quantize_scalar(
            rotated_data, padded_dim, bits, code, delta, vl, config
        );
        if(bits == 8) {
            float* factor = (float*)(code + padded_dim);
            factor[0] = delta;
            factor[1] = std::accumulate(code, code + padded_dim, 0ULL);
        } else {
            throw std::runtime_error("Error: Unsupported bits number");
        }

        delete []rotated_data;
    }

    void getRaBitQCode_l2(float* data, uint8_t* code, size_t bits) {
        float* rotated_data = new float[padded_dim];
        rotator->rotate(data, rotated_data);
        float delta, vl;
        rabitqlib::quant::quantize_scalar(
            rotated_data, padded_dim, bits, code, delta, vl, config
        );
        if(bits == 8) {
            float* factor = (float*)(code + padded_dim);
            factor[0] = delta;
            factor[1] = std::accumulate(code, code + padded_dim, 0ULL);
            factor[2] = std::accumulate(code, code + padded_dim, 0ULL, [](int s, int x) {
                return s + x*x;
            });
        } else {
            throw std::runtime_error("Error: Unsupported bits number");
        }

        delete []rotated_data;
    }

    void InsertPoint(id_t id, size_t efC, float* vec) {
        uint8_t* code = new uint8_t[4*padded_dim + 12];
        if(metric == L2_RaBitQ) {
            getRaBitQCode_l2(vec, code, b_bits);
        } else if(metric == IP_RaBitQ) {
            getRaBitQCode_ip(vec, code, b_bits);
        }

        hnsw_ngfix->InsertPoint(id, efC, code);
        delete[] code;
    }

    void AKNNGroundTruth(float* query, int* gt, size_t k, size_t efC) {
        size_t ndc = 0;
        auto result = searchKnn(query, k, efC, ndc);
        for(int i = 0; i < k; ++i) {
            gt[i] = result[i].second;
        }
    }

    void NGFix(float* query, int* gt, size_t Nq = 100, size_t Kh = 100) {
        uint8_t* query_code = new uint8_t[4*padded_dim + 12];
        if(metric == L2_RaBitQ) {
            getRaBitQCode_l2(query, query_code, q_bits);
        } else if(metric == IP_RaBitQ) {
            getRaBitQCode_ip(query, query_code, q_bits);
        }
        hnsw_ngfix->NGFix(query_code, gt, Nq, Kh);

        delete[] query_code;
    }

    void RFix(float* query, int* gt, size_t Nq = 100, size_t efC = 1500) {
        uint8_t* query_code = new uint8_t[4*padded_dim + 12];
        if(metric == L2_RaBitQ) {
            getRaBitQCode_l2(query, query_code, q_bits);
        } else if(metric == IP_RaBitQ) {
            getRaBitQCode_ip(query, query_code, q_bits);
        }
        hnsw_ngfix->RFix(query_code, gt, Nq, efC);

        delete[] query_code;
    }

    std::vector<std::pair<float, id_t> > searchKnn(float* query_data, size_t k, size_t ef, size_t& ndc) {
        uint8_t* query_code = new uint8_t[4*padded_dim + 12];
        if(metric == L2_RaBitQ) {
            getRaBitQCode_l2(query_data, query_code, q_bits);
        } else if(metric == IP_RaBitQ) {
            getRaBitQCode_ip(query_data, query_code, q_bits);
        }
        
        size_t k2 = k;
        if(is_rerank) {
            k2 = std::min(2*ef, (size_t)(rerank_ratio * k));
        }
        auto res = hnsw_ngfix->searchKnn(query_code, k2, ef, ndc);

        // rerank
        if(k2 > k) {
            for(int i = 0; i < k2; ++i) {
                res[i].first = rerank_space->dist_func(rawdata + res[i].second*dim, query_data);
            }
            std::sort(res.begin(), res.end());
            return std::vector<std::pair<float, id_t> >(res.begin(), res.begin() + k);
        }
        
        delete[] query_code;
        return res;
    }
};
}