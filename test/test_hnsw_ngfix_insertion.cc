#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    size_t efC = 0, insert_st_id = 0;
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_data_path")
            paths["base_data_path"] = argv[i + 1];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
        if (arg == "--raw_index_path")
            paths["raw_index_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_index_path")
            paths["result_index_path"] = argv[i + 1];
        if (arg == "--efC")
            efC = std::stoi(argv[i + 1]);
        if (arg == "--insert_st_id")
            insert_st_id = std::stoi(argv[i + 1]);
        
    }

    std::string base_path = paths["base_data_path"];
    std::cout<<"base_data_path: "<<base_path<<"\n";
    std::string train_query_path = paths["train_query_path"];
    std::cout<<"train_query_path: "<<train_query_path<<"\n";
    std::string train_gt_path = paths["train_gt_path"];
    std::cout<<"train_gt_path: "<<train_gt_path<<"\n";
    std::string base_index_path = paths["raw_index_path"];
    std::cout<<"raw_index_path: "<<base_index_path<<"\n";
    std::string result_index_path = paths["result_index_path"];
    std::cout<<"result_index_path: "<<result_index_path<<"\n";
    std::string metric_str = paths["metric"];

    std::cout<<"efC: "<<efC<<"\n";
    std::cout<<"insert_st_id: "<<insert_st_id<<"\n";

    size_t train_number = 0, base_number = 0;
    size_t train_gt_dim = 0, vecdim = 0;

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

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, base_index_path);

    std::cout << "Raw Index Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    hnsw_ngfix->resize(base_number);

    auto start = std::chrono::high_resolution_clock::now();

    // first insert vectors into base graph
    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = insert_st_id; i < base_number; ++i) {
        if(i % 100000 == 0) {
            std::cout <<"add base points "<< i <<"\n";
        }
        hnsw_ngfix->InsertPoint(i, efC, base_data + i*vecdim);
    }

    // partial rebuilding r
    float r = 0.2;
    hnsw_ngfix->PartialRemoveEdges(r);
    auto train_query_in = getVectorsHead(train_query_path, train_number, vecdim);
    auto train_gt_in = getVectorsHead(train_gt_path, train_number, train_gt_dim); // train_gt_dim >= S
    train_number = train_number*r;
    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 0; i < train_number; ++i) {
        if(i % 100000 == 0) {
            std::cout <<"train queries "<< i <<"\n";
        }
        auto train_query = getNextVector<float>(train_query_in, vecdim);
        auto train_gt = getNextVector<int>(train_gt_in, train_gt_dim);
        hnsw_ngfix->NGFix(train_query, train_gt, 100, 100);
        hnsw_ngfix->NGFix(train_query, train_gt, 10, 10);
        hnsw_ngfix->RFix(train_query, train_gt, 10);
        delete []train_query;
        delete []train_gt; 
    }

    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insertion latency: " << diff << " ms.\n\n";

    std::cout << "Index (after insertion) Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    hnsw_ngfix->StoreIndex(result_index_path);
    return 0;
}