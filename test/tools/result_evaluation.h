#pragma once

#include <vector>
#include <cstring>
#include <string>
#include <map>
#include <cmath>
#include <iostream>
#include <set>
#include <sys/time.h>
#include <unordered_set>
#include "../../ngfixlib/graph/hnsw_ngfix.h"
using namespace ngfixlib;

#define OutputResult

struct SearchResult
{
    float recall;
    size_t ndc;
    int64_t latency;
    double rderr;
};


void AllQueriesEvaluation(std::vector<SearchResult> results, double& avg_recall, double& avg_ndc, double& avg_latency, double& avg_rderr)
{
    int test_number = results.size();

    for(int i = 0; i < test_number; ++i){
        avg_recall += results[i].recall;
        avg_ndc += results[i].ndc;
        avg_rderr += results[i].rderr;
        avg_latency += results[i].latency;
    }
    avg_rderr /= test_number;
    avg_recall /= test_number;
    avg_latency /= test_number;
    avg_latency /= 1000;
    avg_ndc /= test_number;

    #ifdef OutputResult
        std::cerr<<"average recall: "<<avg_recall<<"\n";
        std::cerr<<"average latency: "<<avg_latency<<"ms\n";
        std::cerr<<"average distance computation: "<<avg_ndc<<"\n";
        std::cerr<<"relative distance error: "<<avg_rderr<<"\n";
        std::cerr<<"--------------------------------------\n";
    #endif
}

template<typename T>
SearchResult TestSingleQuery(T* query_data, int* gt, size_t k, size_t efs, HNSW_NGFix<T>* searcher) {
    size_t ndc = 0;

    const unsigned long Converter = 1000 * 1000;
    struct timeval val;
    int ret = gettimeofday(&val, NULL);

    auto aknns = searcher->searchKnn(query_data, k, efs, ndc);
    
    struct timeval newVal;
    ret = gettimeofday(&newVal, NULL);
    int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

    std::unordered_set<id_t> gtset;
    for(int i = 0; i < k; ++i) {
        gtset.insert(gt[i]);
    }

    int acc = 0;
    for(int i = 0; i < k; ++i) {
        if(gtset.find(aknns[i].second) != gtset.end()) {
            ++acc;
            gtset.erase(aknns[i].second);
        }
    }

    float recall = (float)acc/k;
    double rderr = 0;

    for(int i = 0; i < k; ++i){
        float d0 = searcher->getDist(aknns[i].second, query_data);
        float d1 = searcher->getDist(gt[i], query_data);
        if(fabs(d1) < 0.00001) {continue; }
        rderr += d0/d1;
    }

    return SearchResult{recall, ndc, diff, rderr/k};
}

template<typename T>
void TestQueries(std::ostream& s, T* test_query, int* test_gt, size_t test_number, size_t k,
                            size_t test_gt_d, size_t vecdim, HNSW_NGFix<T>* searcher)
{
    //output header
    s << "efs, recall, ndc, latency, rderr\n";
    
    std::vector<int> efss;
    if(k == 100) {
        efss = std::vector<int>{100,110,120,130,140,150,160,180,200,250,300,400,500,800,1000,2000};
    } else if(k == 10) {
        efss = std::vector<int>{10,15,20,30,40,50,60,70,80,90,100,120,150,180,200,300,400,500,800,1000};
    }

    for(auto efs : efss) {
        std::cerr<<efs<<"\n";

        std::vector<SearchResult> results(test_number);
        for(int i = 0; i < test_number; ++i){
            auto gt = test_gt + i*test_gt_d;
            auto res = TestSingleQuery<T>(test_query+1ll*i*vecdim, gt, k, efs, searcher);
            results[i] = {res.recall, res.ndc, res.latency, res.rderr};
        }
        double avg_recall = 0, avg_ndc = 0, avg_latency = 0, avg_rderr = 0;
        AllQueriesEvaluation(results, avg_recall, avg_ndc, avg_latency, avg_rderr);
        s << efs << ", "<< avg_recall << ", " << avg_ndc << ", " << avg_latency << ", " << avg_rderr << "\n";
    }
}
