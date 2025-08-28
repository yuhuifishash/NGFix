#pragma once

#include <memory>
#include <atomic>
#include <shared_mutex>
#include "node.h"
#include "../utils/search_queue.h"
#include "../utils/visited_list.h"
#include "../metric/l2.h"
#include "../metric/ip.h"


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

    void LoadIndex(std::string path) {
        std::ifstream input(path, std::ios::binary);
        input.read((char*)&M, sizeof(M));
        input.read((char*)&M0, sizeof(M0));
        input.read((char*)&MEX, sizeof(MEX));
        input.read((char*)&n, sizeof(n));
        input.read((char*)&entry_point, sizeof(entry_point));

        for(int i = 0; i < n; ++i) {
            Graph[i].LoadIndex(input);
        }

    }

public:
    std::vector<node> Graph;
    T* vecdata;
    Space<T>* space;
    size_t dim;
    size_t entry_point = 0;
    size_t max_elements;
    std::atomic<size_t> n = 0;
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    HNSW_NGFix(Metric metric, size_t dimension, size_t max_elements, T* data, size_t M_ = 16)
                : node_locks(max_elements), M(M_) {
        M0 = 2*M;
        this->dim = dimension;
        this->vecdata = data;
        this->max_elements = max_elements;
        if (metric == L2_float) {
            space = new L2Space_float(dim);
        } else if(metric == IP_float) {
            space = new IPSpace_float(dim);
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, this->max_elements));
    }

    HNSW_NGFix(Metric metric, size_t dimension, size_t max_elements, T* data, std::string path)
                : node_locks(max_elements) {
        this->dim = dimension;
        this->vecdata = data;
        this->max_elements = max_elements;
        if (metric == L2_float) {
            space = new L2Space_float(dim);
        } else if(metric == IP_float) {
            space = new IPSpace_float(dim);
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, this->max_elements));
        LoadIndex(path);
    }

    ~HNSW_NGFix() {
        delete space;
    }

    void resize(T* data, size_t new_max_elements) {
        if(new_max_elements <= this->max_elements) {
            return;
        }
        this->vecdata = data;
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

        for(int i = 0; i < n; ++i) {
            Graph[i].StoreIndex(output);
        }
    }

    T* getData(id_t u) {
        return vecdata + u*dim;
    }

    // <neighbors, out-degree>
    std::pair<id_t*, id_t> getNeighbors(id_t u) {
        auto tmp = Graph[u].get_neighbors();
        return {tmp, GET_SZ((uint8_t*)tmp)};
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
        auto res = searchKnn(data, efC, efC, NDC);
        auto neighbors = getNeighborsByHeuristic(res, M);
        
        { // add edge (cur_id, neighbor_id)
            std::unique_lock <std::shared_mutex> lock(node_locks[cur_id]);
            Graph[cur_id].replace_base_graph_neighbors(neighbors);
        }

        // add edge (neighbor_id, cur_id)
        for(auto [_, neighbor_id] : neighbors) {
            std::unique_lock <std::shared_mutex> lock(node_locks[neighbor_id]);
            auto [ids, sz] = getNeighbors(neighbor_id);
            if(sz < M0) {
                Graph[neighbor_id].add_base_graph_neighbors(cur_id);
            } else {
                // finding the "weakest" element to replace it with the new one
                float d_max = getDist(neighbor_id, cur_id);
                // Heuristic:
                std::vector<std::pair<float, id_t> > candidates;
                candidates.push_back({d_max, cur_id});
                for (size_t j = 1; j <= sz; j++) {
                    candidates.push_back({getDist(ids[j], neighbor_id) , ids[j]});
                }
                std::sort(candidates.begin(), candidates.end());
                auto neighbors = getNeighborsByHeuristic(candidates, M0);

                Graph[neighbor_id].replace_base_graph_neighbors(neighbors);
            }
        }
        

    }
    void PrepareData(T* data) {
        vecdata = data;
    }

    // The insertion method is HNSW's bottom layer
    void InsertPoint(id_t id, size_t efC) {
        if(id >= max_elements) {
            throw std::runtime_error("Error: id > max_elements.");
        }
        auto data = getData(id);

        if(n != 0) {
            HNSWBottomLayerInsertion(data, id, efC);
        }

        ++n;
        if(n % 100000 == 0) {
            SetEntryPoint();
        }
    }

    void DeletePoint(id_t id) {

    }

    // set ep to centroid
    void SetEntryPoint() {
        T* centroid = new T[dim];
        memset(centroid, 0, sizeof(T)*dim);
        for(int i = 0; i < n; ++i) {
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

    std::vector<std::pair<float, id_t> > searchKnn(T* query_data, size_t k, size_t ef, size_t& ndc) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        Search_PriorityQueue q0(ef);
        auto q = &q0;

        float dist = getDist(entry_point, query_data);
        q->push(entry_point, dist);
        visited_array[entry_point] = visited_array_tag;
        while (!q->is_empty()) {
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            _mm_prefetch((char *)(visited_array + current_node_id), _MM_HINT_T0);
            _mm_prefetch((char *)(vecdata + current_node_id * dim), _MM_HINT_T0);

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }
            
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz] = getNeighbors(current_node_id);
            for (int i = 1; i <= sz; ++i) {
                int candidate_id = outs[i];

                if(i < sz) {
                    _mm_prefetch((char*) (visited_array + outs[i+1]), _MM_HINT_T0);
                    _mm_prefetch((char*) (vecdata + outs[i+1] * dim), _MM_HINT_T0);
                }

                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    float dist = getDist(candidate_id, query_data);
                    q->push(candidate_id, dist);

                    ndc += 1;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        auto res = q->get_result(k);

        return res;
    }

    void printGraphInfo() {
        double avg_outdegree = 0;
        double avg_capacity = 0;

        std::cout << "current number of elements: " << n << "\n";

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