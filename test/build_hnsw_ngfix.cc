#include "../ngfixlib/graph/hnsw_ngfix.h"
#include "data_loader.h"
#include "result_evaluation.h"
#include <iostream>
using namespace ngfixlib;

int main()
{
    size_t test_number = 0, base_number = 0, train_number = 0;
    size_t test_gt_dim = 0, train_gt_dim = 0, vecdim = 0;
    size_t efC = 2000, M = 16, MEX = 48;

    std::string data_path = "/SSD/WebVid/";
    auto test_query = LoadData<float>(data_path + "webvid.query.10k.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "gt.query.top100.bin", test_number, test_gt_dim);
    auto base_data = LoadData<float>(data_path + "webvid.base.2.5M.fbin", base_number, vecdim);

    // std::string data_path = "/SSD/SIFT1M/";
    // auto test_query = LoadData<float>(data_path + "sift_query.fbin", test_number, vecdim);
    // auto test_gt = LoadData<int>(data_path + "gt.query.top100.bin", test_number, test_gt_dim);
    // auto base_data = LoadData<float>(data_path + "sift_base.fbin", base_number, vecdim);

    // std::string data_path = "/SSD/DEEP10M/";
    // auto test_query = LoadData<float>(data_path + "query.fbin", test_number, vecdim);
    // auto test_gt = LoadData<int>(data_path + "gt.query.top100.bin", test_number, test_gt_dim);
    // auto base_data = LoadData<float>(data_path + "base.fbin", base_number, vecdim);

    // std::string data_path = "/SSD/SIFT1B/";
    // auto test_query = LoadData<float>(data_path + "query.public.10K.fbin", test_number, vecdim);
    // auto test_gt = LoadData<int>(data_path + "gt.public.10K_10M_top100.bin", test_number, test_gt_dim);
    // auto u8_base = LoadData<uint8_t>(data_path + "slice.10M.u8bin", base_number, vecdim);
    // auto base_data = u8_f32(u8_base, base_number*vecdim);
    // delete []u8_base;
    
    // std::string data_path = "/SSD/Text-to-Image/";
    // auto test_query = LoadData<float>(data_path + "query.10k.fbin", test_number, vecdim);
    // auto test_gt = LoadData<int>(data_path + "gt.10K_10M.bin", test_number, test_gt_dim);
    // auto base_data = LoadData<float>(data_path + "base.10M.fbin", base_number, vecdim);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto hnsw_ngfix = new HNSW_NGFix<float>(IP_float, vecdim, base_number, base_data, "/SSD/models/hnsw/webvid_bottom_layer");
    // auto hnsw_ngfix = new HNSW_NGFix<float>(IP_float, vecdim, base_number, base_data, M, MEX);
    // hnsw_ngfix->InsertPoint(0, efC);

    // #pragma omp parallel for schedule(dynamic) num_threads(32)
    // for(int i = 1; i < base_number; ++i) {
    //     // printf("\n======================== %d\n",i);
    //     if(i % 100000 == 0) {
    //         std::cout <<"add base points "<< i <<"\n";
    //     }
    //     hnsw_ngfix->InsertPoint(i, efC);
    // }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "HNSW Bottom Layer construction latency: " << diff << " ms.\n\n";

    // hnsw_ngfix->StoreIndex("/SSD/models/hnsw/webvid_bottom_layer");

    std::cout << "HNSW Bottom Layer Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    auto train_query_in = getVectorsHead(data_path + "webvid.query.train.2.5M.fbin", train_number, vecdim);
    auto train_gt_in = getVectorsHead(data_path + "gt.train.top500.bin", train_number, train_gt_dim); // train_gt_dim >= S

    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 0; i < train_number; ++i) {
        if(i % 100000 == 0) {
            std::cout <<"train queries "<< i <<"\n";
        }
        auto train_query = getNextVector<float>(train_query_in, vecdim);
        auto train_gt = getNextVector<int>(train_gt_in, train_gt_dim);
        hnsw_ngfix->NGFix(train_query, train_gt, 100, 100);
        hnsw_ngfix->NGFix(train_query, train_gt, 10, 10);
        delete []train_query;
        delete []train_gt;
    }

    
    end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "NGFix latency: " << diff << " ms.\n\n";

    std::cout << "HNSW_NGFix Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";
    hnsw_ngfix->StoreIndex("/SSD/models/hnsw/hnsw_ngfix_webvid");

    std::ofstream output;
    output.open("/home/hzy/NGFix2/test/ngfix.csv");
    TestQueries<float>(output, test_query, test_gt, test_number, 100, test_gt_dim, vecdim, hnsw_ngfix);

    return 0;
}