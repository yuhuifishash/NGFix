#include "ngfixlib/graph/hnsw_ngfix_rabitq.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation_rabitq.h"
#include <iostream>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_data_path")
            paths["base_data_path"] = argv[i + 1];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
        if (arg == "--base_graph_path")
            paths["base_graph_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_index_path")
            paths["result_index_path"] = argv[i + 1];
        
    }

    std::string base_path = paths["base_data_path"];
    std::cout<<"base_data_path: "<<base_path<<"\n";
    std::string train_query_path = paths["train_query_path"];
    std::cout<<"train_query_path: "<<train_query_path<<"\n";
    std::string train_gt_path = paths["train_gt_path"];
    std::cout<<"train_gt_path: "<<train_gt_path<<"\n";
    std::string base_index_path = paths["base_graph_path"];
    std::cout<<"base_graph_path: "<<base_index_path<<"\n";
    std::string result_index_path = paths["result_index_path"];
    std::cout<<"result_index_path: "<<result_index_path<<"\n";
    std::string metric_str = paths["metric"];


    size_t train_number = 0;
    size_t train_gt_dim = 0, vecdim = 0;

    Metric metric;
    if(metric_str == "ip_rabitq") {
        std::cout<<"metric ip\n";
        metric = IP_RaBitQ;
    } else if(metric_str == "l2_rabitq") {
        std::cout<<"metric l2\n";
        metric = L2_RaBitQ;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }

    auto hnsw_ngfix_rabitq = new HNSW_NGFix_RaBitQ(metric, base_index_path);

    std::cout << "HNSW Bottom Layer Information:\n";
    hnsw_ngfix_rabitq->hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    auto train_query_in = getVectorsHead(train_query_path, train_number, vecdim);
    auto train_gt_in = getVectorsHead(train_gt_path, train_number, train_gt_dim); // train_gt_dim >= S

    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 0; i < train_number; ++i) {
        if(i % 100000 == 0) {
            std::cout <<"train queries "<< i <<"\n";
        }
        auto train_query = getNextVector<float>(train_query_in, vecdim);
        auto train_gt = getNextVector<int>(train_gt_in, train_gt_dim);
        hnsw_ngfix_rabitq->NGFix(train_query, train_gt, 100, 100);
        hnsw_ngfix_rabitq->NGFix(train_query, train_gt, 10, 10);
        hnsw_ngfix_rabitq->RFix(train_query, train_gt, 10);
        delete []train_query;
        delete []train_gt; 
    }

    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "NGFix latency: " << diff << " ms.\n\n";

    std::cout << "HNSW_NGFix Information:\n";
    hnsw_ngfix_rabitq->hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    hnsw_ngfix_rabitq->StoreIndex(result_index_path);
    return 0;
}