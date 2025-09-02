#pragma once

#include <memory>
#include <atomic>
#include <shared_mutex>
#include <bitset>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <map>
#include "node.h"
#include "../utils/search_list.h"
#include "../utils/visited_list.h"
#include "../metric/l2.h"
#include "../metric/ip.h"

static const size_t MAX_Nq = 500;
static const size_t MAX_S = 500;

namespace ngfixlib {

enum Metric{
    L2_float = 1,
    IP_float = 2,
    L2_uint8 = 3,
    IP_uint8 = 4
};

template<typename T>
class HNSW_NGFix
{
private:
    // lock when updating neighbors
    std::vector<std::shared_mutex> node_locks;
    size_t M0 = 0; // out-degree of base layer  
    size_t M = 0;  // used for insertion
    size_t MEX = 0;

public:
    std::vector<node> Graph;
    char* vecdata;
    Space<T>* space;
    size_t dim;
    size_t entry_point = 0;
    size_t max_elements;
    std::atomic<size_t> n = 0;
    std::shared_ptr<VisitedListPool> visited_list_pool_{nullptr};
    
    std::shared_mutex delete_lock;
    std::unordered_set<id_t> delete_ids;

    size_t size_per_element = 0;

    HNSW_NGFix(Metric metric, size_t dimension, size_t max_elements, size_t M_ = 16, size_t MEX_ = 48)
                : node_locks(max_elements), M(M_), MEX(MEX_) {
        M0 = 2*M;
        this->dim = dimension;
        this->max_elements = max_elements;
        this->size_per_element = dim*sizeof(T) + 1; // 8 bits for delete flag
        if (metric == L2_float) {
            space = new L2Space_float(dim);
        } else if(metric == IP_float) {
            space = new IPSpace_float(dim);
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }
        vecdata = new char[size_per_element*max_elements];
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::shared_ptr<VisitedListPool>(new VisitedListPool(1, this->max_elements));
    }

    HNSW_NGFix(Metric metric, std::string path) {
        std::ifstream input(path, std::ios::binary);

        input.read((char*)&M, sizeof(M));
        input.read((char*)&M0, sizeof(M0));
        input.read((char*)&MEX, sizeof(MEX));
        input.read((char*)&n, sizeof(n));
        input.read((char*)&entry_point, sizeof(entry_point));
        input.read((char*)&dim, sizeof(dim));
        input.read((char*)&max_elements, sizeof(max_elements));

        this->size_per_element = dim*sizeof(T) + 1; // 8 bits for delete flag
        if (metric == L2_float) {
            space = new L2Space_float(dim);
        } else if(metric == IP_float) {
            space = new IPSpace_float(dim);
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }
        vecdata = new char[size_per_element*max_elements];
        input.read(vecdata, max_elements*size_per_element);

        std::vector<std::shared_mutex>(this->max_elements).swap(node_locks);
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::shared_ptr<VisitedListPool>(new VisitedListPool(1, this->max_elements));
        
        for(int i = 0; i < n; ++i) {
            Graph[i].LoadIndex(input);
        }
    }

    ~HNSW_NGFix() {
        delete space;
    }

    void resize(size_t new_max_elements) {
        if(new_max_elements <= this->max_elements) {
            return;
        }
        
        auto new_vecdata = new char[size_per_element*new_max_elements];
        memcpy(new_vecdata, vecdata, size_per_element*max_elements);        
        delete []vecdata;
        vecdata = new_vecdata;

        this->max_elements = new_max_elements;
        std::vector<std::shared_mutex>(new_max_elements).swap(node_locks);
        Graph.resize(new_max_elements);
        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));
    }

    void StoreIndex(std::string path) {
        std::ofstream output(path, std::ios::binary);

        output.write((char*)&M, sizeof(M));
        output.write((char*)&M0, sizeof(M0));
        output.write((char*)&MEX, sizeof(MEX));
        output.write((char*)&n, sizeof(n));
        output.write((char*)&entry_point, sizeof(entry_point));
        output.write((char*)&dim, sizeof(dim));
        output.write((char*)&max_elements, sizeof(max_elements));

        output.write(vecdata, max_elements*size_per_element);
        for(int i = 0; i < n; ++i) {
            Graph[i].StoreIndex(output);
        }
    }

    T* getData(id_t u) {
        return (T*)(vecdata + u*size_per_element + 1);
    }

    void SetData(id_t u, T* data) {
        memcpy(getData(u), data, sizeof(T)*dim);
    }

    // <neighbors, out-degree>
    auto getNeighbors(id_t u) {
        auto tmp = Graph[u].get_neighbors();
        return std::tuple{tmp, GET_SZ((uint8_t*)tmp), GET_NGFIX_CAPACITY((uint8_t*)tmp)-GET_NGFIX_SZ((uint8_t*)tmp) + 1};
    }

    auto getBaseGraphNeighbors(id_t u) {
        auto tmp = Graph[u].get_neighbors();
        return std::tuple{tmp, GET_SZ((uint8_t*)tmp)-GET_NGFIX_SZ((uint8_t*)tmp), GET_NGFIX_CAPACITY((uint8_t*)tmp) + 1};
    }

    float getDist(id_t u, id_t v) {
        return space->dist_func(getData(u), getData(v));
    }

    float getDist(id_t u, float* query_data) {
        return space->dist_func(getData(u), query_data);
    }


    std::vector<std::pair<float, id_t> > 
    getNeighborsByHeuristic(std::vector<std::pair<float, id_t> >& neighbor_candidates, const size_t M) {
        if (neighbor_candidates.size() < M) {
            return neighbor_candidates;
        }

        std::vector<std::pair<float, id_t> > return_list;

        for (auto [dist_to_query, cur_id] : neighbor_candidates) {
            if (return_list.size() >= M) {break;}
            bool good = true;

            for (auto [_, exist_id] : return_list) {
                float curdist = getDist(exist_id, cur_id);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back({dist_to_query, cur_id});
            }
        }

        return return_list;
    }


    void HNSWBottomLayerInsertion(T* data, id_t cur_id, size_t efC) {
        size_t NDC = 0;
        auto res = searchKnnBaseGraph(data, efC, efC, NDC);
        auto neighbors = getNeighborsByHeuristic(res, M);
        
        { // add edge (cur_id, neighbor_id)
            std::unique_lock <std::shared_mutex> lock(node_locks[cur_id]);
            Graph[cur_id].replace_base_graph_neighbors(neighbors);
        }

        // add edge (neighbor_id, cur_id)
        for(auto [_, neighbor_id] : neighbors) {
            std::unique_lock <std::shared_mutex> lock(node_locks[neighbor_id]);
            auto [ids, sz, st] = getBaseGraphNeighbors(neighbor_id);
            if(sz < M0) {
                Graph[neighbor_id].add_base_graph_neighbors(cur_id);
            } else {
                // finding the "weakest" element to replace it with the new one
                float d_max = getDist(neighbor_id, cur_id);
                // Heuristic:
                std::vector<std::pair<float, id_t> > candidates;
                candidates.push_back({d_max, cur_id});
                for (int j = st; j < st + sz; j++) {
                    candidates.push_back({getDist(ids[j], neighbor_id) , ids[j]});
                }
                std::sort(candidates.begin(), candidates.end());
                auto neighbors = getNeighborsByHeuristic(candidates, M0);

                Graph[neighbor_id].replace_base_graph_neighbors(neighbors);
            }
        }
    }

    // The insertion method is HNSW's bottom layer
    void InsertPoint(id_t id, size_t efC, T* vec) {
        if(id >= max_elements) {
            throw std::runtime_error("Error: id > max_elements.");
        }
        SetData(id, vec);
        auto data = getData(id);

        if(n != 0) {
            HNSWBottomLayerInsertion(data, id, efC);
        }
        ++n;
        if(n % 100000 == 0) {
            SetEntryPoint();
        }
    }

    // remove r% NGFix edges
    void PartialRemoveEdges(float r) {
        for(int i = 0; i < n; ++i) {
            std::unique_lock <std::shared_mutex> lock(node_locks[i]);
            auto neighbors = Graph[i].neighbors;
            std::vector<id_t> new_neighbors;
            uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
            uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
            if(ngfix_sz == 0) {
                continue;
            } else {
                int st = ngfix_capacity - ngfix_sz;
                for(int j = st; j < std::max(1, (int32_t)(ngfix_sz*(1.0 - r))) + st; ++j) {
                    new_neighbors.push_back(neighbors[j + 1]);
                }
            }
            Graph[i].replace_ngfix_neighbors(new_neighbors);
        }
    }

    void set_deleted(id_t id) {
        (vecdata + id*size_per_element)[0] = true;
    }
    bool is_deleted(id_t id) {
        return (vecdata + id*size_per_element)[0];
    }

    // You can replace the data corresponding to the deleted ID with a new vector and re-insert it after calling DeleteAllFlagNodesByNGFix.
    void DeleteAllFlagPointsByNGFix(size_t efC_delete = 500, size_t Threads = 32) {
        std::unordered_set<id_t> ids;
        {
            std::unique_lock <std::shared_mutex> lock(delete_lock);
            ids = delete_ids;
            delete_ids.clear();
        }

        // delete correspoding edges
        #pragma omp parallel for schedule(dynamic) num_threads(Threads)
        for(int i = 0; i < n; ++i) {
            if(ids.find(i) != ids.end()) {
                Graph[i].delete_node();
            } else {
                // replace ngfix neighbors and base graph neighbors
                auto [outs, sz, st] = getNeighbors(i);
                uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)outs);
                uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)outs);
                uint8_t base_sz = sz - ngfix_sz;

                std::vector<id_t> new_neighbors;
                for(int i = ngfix_capacity - ngfix_sz; i < ngfix_capacity; ++i) {
                    if(ids.find(outs[i + 1]) == ids.end()) { // not deleted
                        new_neighbors.push_back(outs[i + 1]);
                    }
                }
                Graph[i].replace_ngfix_neighbors(new_neighbors);
                new_neighbors.clear();

                for(int i = ngfix_capacity; i < ngfix_capacity + base_sz; ++i) {
                    if(ids.find(outs[i + 1]) == ids.end()) { // not deleted
                        new_neighbors.push_back(outs[i + 1]);
                    }
                }
                Graph[i].replace_base_graph_neighbors(new_neighbors);
            }
        }

        std::vector<id_t> v_ids;
        for(auto u : ids) {
            v_ids.push_back(u);
        }

        #pragma omp parallel for schedule(dynamic) num_threads(Threads)
        for(int i = 0; i < v_ids.size(); ++i) {
            int* gt = new int[500];
            AKNNGroundTruth(getData(v_ids[i]), gt, 500, efC_delete);
            NGFix(getData(v_ids[i]), gt, 100, 100);
            delete []gt; 
        }
    }

    // This function is not thread-safe.
    // We only guarantee it can be called concurrently with DeleteAllFlagPointsByNGFix.
    void DeletePointByFlag(id_t id) {
        set_deleted(id);
        {
            std::shared_lock <std::shared_mutex> lock(delete_lock);
            if(id != entry_point) {
                delete_ids.insert(id);
                n = n - 1;
            }
        }
    }

    std::unordered_map<id_t, std::vector<id_t> > ComputeGq(int* gt, size_t S) 
    {
        std::unordered_map<id_t, std::vector<id_t> > G;
        std::unordered_set<id_t> Vq;
        for(int i = 0; i < S; ++i){
            Vq.insert(gt[i]);
        }
        for(int i = 0; i < S; ++i){
            int u = gt[i];
            auto [ids, sz, st] = getNeighbors(u);
            for (int j = st; j < st + sz; ++j){
                id_t v = ids[j];
                if(Vq.find(v) == Vq.end()){
                    continue;
                }
                G[u].push_back(v);
            }
        }

        return G;
    }

    std::vector<std::vector<uint16_t> > CalculateHardness(int* gt, size_t Nq, size_t Kh, size_t S) 
    {
        auto Gq = ComputeGq(gt, S);
        std::unordered_map<id_t, uint16_t> p2rank;
        for(int i = 0; i < S; ++i){
            p2rank[gt[i]] = i;
        }
        
        std::vector<std::vector<uint16_t> > H;
        std::bitset<MAX_S> f[S];
        H.resize(Nq);
        for(int i = 0; i < Nq; ++i){
            H[i].resize(Nq, EH_INF);
        }
        for(int h = 0; h < S; ++h){
            f[h][h] = 1;
            if(h < Nq){
                H[h][h] = h;
            }
        }

        for(auto [u, neighbors] : Gq){
            int i = p2rank[u];
            for(auto v : neighbors){
                int j = p2rank[v];
                f[i][j] = 1;
                if(i < Nq && j < Nq){
                    H[i][j] = std::max(i,j);
                }
            }
        }

        for(int h = 0; h < S; ++h){
            for(int i = 0; i < S; ++i){
                auto last = f[i];
                if(f[i][h]){
                    f[i] |= f[h];
                }
                last ^= f[i];
                if(i < Nq && last.count() > 0){
                    for(int j = 0; j < Nq; ++j){
                        if(last[j] == 1){
                            H[i][j] = h;
                        }
                    }
                }
            }
        }
        return H;
    }

    auto getDefectsFixingEdges(
        std::bitset<MAX_Nq> f[],
        std::vector<std::vector<uint16_t> >& H,
        float* query, int* gt, size_t Nq, size_t Kh) {
        
        std::unordered_map<id_t, std::vector<std::pair<id_t, uint16_t> > > new_edges;

        std::vector<std::pair<float, std::pair<int,int> > > vs;
        for(int i = 0; i < Nq; ++i){
            for(int j = 0; j < Nq; ++j){
                if(f[i][j] == 1) {continue;}
                int u = gt[i];
                int v = gt[j]; 
                float d = getDist(u, v);
                vs.push_back({d,{i,j}});
            }
        }
        std::sort(vs.begin(), vs.end());

        for(auto [d, e] : vs){
            int s = e.first;
            int t = e.second;
            if(f[s][t] == 1) {continue;}

            int u = gt[s];
            int v = gt[t];

            new_edges[u].push_back({v, H[s][t]});

            f[s][t] = 1;
            for(int i = 0; i < Nq; ++i){
                if(f[i][s]){
                    f[i] |= f[t];
                }
            }
        }
        return new_edges;
    }

    void NGFix(T* query, int* gt, size_t Nq = 100, size_t Kh = 100) {
        if(Nq > MAX_Nq) {
            throw std::runtime_error("Error: Nq >= MAX_Nq.");
        }
        auto H = CalculateHardness(gt, Nq, Kh, std::min(MAX_S, 3*Nq));
        std::bitset<MAX_Nq> f[Nq];
        for(int i = 0; i < Nq; ++i){
            for(int j = 0; j < Nq; ++j){
                f[i][j] = (H[i][j] <= Kh) ? 1 : 0;
            }
        }
        auto new_edges = getDefectsFixingEdges(f, H, query, gt, Nq, Kh);
        for(auto [u, vs] : new_edges) {
            std::unique_lock <std::shared_mutex> lock(node_locks[u]);
            for(auto [v, eh] : vs) {
                // printf("add edges %d %d\n", u,v);
                Graph[u].add_ngfix_neighbors(v, eh, MEX);
            }
        }
        
    }

    // return S = {v | delta(v, q) < delta(u, q)}
    std::vector<id_t> searchCloserPoints(T* query_data, size_t ef, size_t u, size_t& ndc) {
        // Search_PriorityQueue q0(ef, visited_list_pool_);
        // Search_Array q0(ef, visited_list_pool_);
        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;
        
        std::vector<id_t> res;
        float dist_u_q = getDist(u, query_data);
        q->push(u, dist_u_q, is_deleted(u));
        q->set_visited(u);
        while (!q->is_empty()) {
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }

            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }
                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getDist(candidate_id, query_data);
                    if(dist < dist_u_q && !is_deleted(candidate_id)) {
                        res.push_back(candidate_id);
                    }
                    q->push(candidate_id, dist, is_deleted(candidate_id));
                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        return res;
    }

    /* Experiments show that for most historical queries, 
       a single round of RF is sufficient to bring the search to the vicinity of the query. 
       For simplicity, we perform only one round of RFix. 
    */
    void RFix(T* query, int* gt, size_t Nq = 10, size_t efC = 1500) {
        size_t ndc = 0;
        size_t k = 1;
        auto result = searchKnn(query, k, Nq, ndc);
        
        id_t ANN = result[0].second;
        float d1 = result[0].first;
        float d2 = getDist(gt[Nq - 1], query);
        if(d1 > d2) { // can not reach NG_{Nq,q}
            auto res = searchCloserPoints(query, efC, ANN, ndc);
            
            id_t u = ANN;
            std::vector<std::pair<float, id_t> > candidates;
            for(auto i : res){
                candidates.push_back({getDist(u, i), i});
            }
            std::sort(candidates.begin(), candidates.end());

            auto neighbors = getNeighborsByHeuristic(candidates, 6);

            {
                std::unique_lock <std::shared_mutex> lock(node_locks[u]);
                for(auto [d, v] : neighbors) {
                    Graph[u].add_ngfix_neighbors(v, MAX_S + 1, MEX);
                }
            }
        }
    }

    void AKNNGroundTruth(T* query, int* gt, size_t k, size_t efC) {
        size_t ndc = 0;
        auto result = searchKnn(query, k, efC, ndc);
        for(int i = 0; i < k; ++i) {
            gt[i] = result[i].second;
        }
    }

    // set ep to centroid
    void SetEntryPoint() {
        T* centroid = new T[dim];
        memset(centroid, 0, sizeof(T)*dim);
        for(int i = 0; i < n; ++i) {
            if(is_deleted(i)) {continue;}
            auto data = getData(i);
            for(int d = 0; d < dim; ++d) {
                centroid[d] += data[d];
            }
        }

        for(int d = 0; d < dim; ++d) {
            centroid[d] /= n;
        }

        id_t ep = 0;
        float min_dis = std::numeric_limits<float>::max();
        for(int i = 0; i < n; ++i) {
            auto dis = space->dist_func(centroid, getData(i));
            if(dis < min_dis) {
                min_dis = dis;
                ep = i;
            }
        }
        entry_point = ep;
        delete []centroid;
    }

    std::vector<std::pair<float, id_t> > searchKnnBaseGraph(T* query_data, size_t k, size_t ef, size_t& ndc) {
        // Search_PriorityQueue q0(ef, visited_list_pool_);
        // Search_Array q0(ef, visited_list_pool_);
        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        float dist = getDist(entry_point, query_data);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        while (!q->is_empty()) {
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }
            
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getBaseGraphNeighbors(current_node_id);

            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }

                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));

                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        auto res = q->get_result(k);

        return res;
    }

    std::vector<std::pair<float, id_t> > searchKnn(T* query_data, size_t k, size_t ef, size_t& ndc) {
        // Search_PriorityQueue q0(ef, visited_list_pool_);
        // Search_Array q0(ef, visited_list_pool_);
        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;
        
        float dist = getDist(entry_point, query_data);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        while (!q->is_empty()) {
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }
            // std::cout<<current_node_id<<" "<<candidate_dist<<"\n";
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }
                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));

                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        auto res = q->get_result(k);
        return res;
    }

    void printGraphInfo() {
        double avg_outdegree = 0;
        double avg_capacity = 0;

        std::cout << "current number of elements: " << n << "\n";
        std::cout << "max number of elements: " << max_elements << "\n";

        for(int i = 0; i < n; ++i) {
            avg_outdegree += GET_SZ((uint8_t*)Graph[i].neighbors);
            avg_capacity += GET_CAPACITY((uint8_t*)Graph[i].neighbors);
        }
        avg_outdegree /= n;
        avg_capacity /= n;
        std::cout << "Average out-degree: " << avg_outdegree << "\n";
        std::cout << "Average Capacity: " << avg_capacity << "\n";
        std::cout << "entry point: " << entry_point << "\n";
    }
};

}