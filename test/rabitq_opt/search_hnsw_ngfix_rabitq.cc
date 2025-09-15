#include "ngfixlib/graph/hnsw_ngfix_rabitq.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation_rabitq.h"
#include <iostream>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    int k = 0;
    bool is_rerank = false;
    size_t q_bits = 0;
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test_query_path")
            paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path")
            paths["test_gt_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--index_path")
            paths["index_path"] = argv[i + 1];
        if (arg == "--result_path")
            paths["result_path"] = argv[i + 1];
        if (arg == "--K")
            k = std::stoi(argv[i + 1]);
        if (arg == "--is_rerank")
            is_rerank = std::stoi(argv[i + 1]);
        if (arg == "--raw_vector_path")
            paths["raw_vector_path"] = argv[i + 1];
        if (arg == "--q_bits")
            q_bits = std::stoi(argv[i + 1]);
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::cout<<"test_query_path: "<<test_query_path<<"\n";
    std::string test_gt_path = paths["test_gt_path"];
    std::cout<<"test_gt_path: "<<test_gt_path<<"\n";
    std::string index_path = paths["index_path"];
    std::cout<<"index_path: "<<index_path<<"\n";
    std::string result_path = paths["result_path"];
    std::cout<<"result_path: "<<result_path<<"\n";
    std::string metric_str = paths["metric"];

    size_t test_number = 0, base_number = 0;
    size_t test_gt_dim = 0, vecdim = 0;

    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
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

    auto hnsw_ngfix_rabitq = new HNSW_NGFix_RaBitQ(metric, index_path);
    hnsw_ngfix_rabitq->hnsw_ngfix->printGraphInfo();
    std::cout<<"q_bits: "<<q_bits<<"\n";
    hnsw_ngfix_rabitq->set_q_bits(q_bits);

    std::cout<<"is_rerank: "<<is_rerank<<"\n";
    if(is_rerank) {
        std::string raw_vector_path = paths["raw_vector_path"];
        std::cout<<"raw_vector_path: "<<raw_vector_path<<"\n";

        auto raw_vector = LoadData<float>(raw_vector_path, base_number, vecdim);
        hnsw_ngfix_rabitq->rerank_init(raw_vector);
    }

    std::ofstream output;
    output.open(result_path);
    TestQueries(output, test_query, test_gt, test_number, k, test_gt_dim, vecdim, hnsw_ngfix_rabitq);

    return 0;
}