#include "ngfixlib/graph/hnsw_ngfix_rabitq.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation_rabitq.h"
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    size_t M = 0, efC = 0, MEX = 0, b_bits = 0;
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
        if (arg == "--b_bits")
            b_bits = std::stoi(argv[i + 1]);
    }

    std::string base_path = paths["base_data_path"];
    std::cout<<"base_data_path: "<<base_path<<"\n";
    std::string result_hnsw_index_path = paths["result_hnsw_index_path"];
    std::cout<<"result_hnsw_index_path: "<<result_hnsw_index_path<<"\n";
    std::string metric_str = paths["metric"];

    std::cout<<"M:"<<M<<"  efC:"<<efC<<"  MEX:"<<MEX<<"  bits:"<<b_bits<<"\n";

    size_t base_number = 0;
    size_t vecdim = 0;

    auto base_data = LoadData<float>(base_path, base_number, vecdim);

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


    auto start = std::chrono::high_resolution_clock::now();

    auto hnsw_ngfix_rabitq = new HNSW_NGFix_RaBitQ(metric, vecdim, base_number, b_bits, M, MEX);
    hnsw_ngfix_rabitq->InsertPoint(0, efC, base_data);

    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 1; i < base_number; ++i) {
        if(i % 100000 == 0) {
            std::cout <<"add base points "<< i <<"\n";
        }
        hnsw_ngfix_rabitq->InsertPoint(i, efC, base_data + i*vecdim);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "HNSW Bottom Layer construction latency: " << diff << " ms.\n\n";

    hnsw_ngfix_rabitq->StoreIndex(result_hnsw_index_path);

    std::cout << "HNSW Bottom Layer Information:\n";
    hnsw_ngfix_rabitq->hnsw_ngfix->printGraphInfo();
    std::cout << "\n";
    return 0;
}