#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    size_t M = 0, efC = 0, MEX = 0;
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_data_path")
            paths["base_data_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_hnsw_index_path")
            paths["result_hnsw_index_path"] = argv[i + 1];
        if (arg == "--efC")
            efC = std::stoi(argv[i + 1]);
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--MEX")
            MEX = std::stoi(argv[i + 1]);
    }

    std::string base_path = paths["base_data_path"];
    std::cout<<"base_data_path: "<<base_path<<"\n";
    std::string result_hnsw_index_path = paths["result_hnsw_index_path"];
    std::cout<<"result_hnsw_index_path: "<<result_hnsw_index_path<<"\n";
    std::string metric_str = paths["metric"];

    std::cout<<"M:"<<M<<"  efC:"<<efC<<"  MEX:"<<MEX<<"\n";

    size_t base_number = 0;
    size_t vecdim = 0;

    auto base_data = LoadData<float>(base_path, base_number, vecdim);

    Metric metric;
    if(metric_str == "ip_float") {
        std::cout<<"metric ip\n";
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        std::cout<<"metric l2\n";
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, vecdim, base_number, M, MEX);
    hnsw_ngfix->InsertPoint(0, efC, base_data);

    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 1; i < base_number; ++i) {
        if(i % 100000 == 0) {
            std::cout <<"add base points "<< i <<"\n";
        }
        hnsw_ngfix->InsertPoint(i, efC, base_data + i*vecdim);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "HNSW Bottom Layer construction latency: " << diff << " ms.\n\n";

    hnsw_ngfix->StoreIndex(result_hnsw_index_path);

    std::cout << "HNSW Bottom Layer Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";
    return 0;
}