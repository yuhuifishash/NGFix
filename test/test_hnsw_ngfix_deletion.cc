#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    int k = 0;
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--index_path")
            paths["index_path"] = argv[i + 1];
        if (arg == "--result_index_path")
            paths["result_index_path"] = argv[i + 1];
        if (arg == "--K")
            k = std::stoi(argv[i + 1]);
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::cout<<"test_query_path: "<<test_query_path<<"\n";
    std::string test_gt_path = paths["test_gt_path"];
    std::cout<<"test_gt_path: "<<test_gt_path<<"\n";
    std::string index_path = paths["index_path"];
    std::cout<<"index_path: "<<index_path<<"\n";
    std::string result_index_path = paths["result_index_path"];
    std::cout<<"result_index_path: "<<result_index_path<<"\n";
    std::string metric_str = paths["metric"];

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

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, index_path);
    std::cout << "Raw Index Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";


    auto start = std::chrono::high_resolution_clock::now();

    for(int i = hnsw_ngfix->max_elements*0.8; i < hnsw_ngfix->max_elements; ++i) {
        hnsw_ngfix->DeletePointByFlag(i);
    }
    hnsw_ngfix->DeleteAllFlagPointsByNGFix();

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Deletion latency: " << diff << " ms.\n\n";
    
    std::cout << "Index (after deletion) Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";
    
    hnsw_ngfix->StoreIndex(result_index_path);

    return 0;
}