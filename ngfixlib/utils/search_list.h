#pragma once
#include "../ngfixlib/graph/node.h"
#include "visited_list.h"
#include <queue>
#include <algorithm>
#include <memory>
#include <xmmintrin.h>

namespace ngfixlib {

class SearchList
{
public:
    virtual float get_dist_bound() = 0;
    virtual void push(id_t id, float dist) = 0;
    virtual std::pair<float, id_t> get_next_id() = 0; // also pop
    virtual bool is_empty() = 0;
    virtual std::vector<std::pair<float, id_t> > get_result(size_t k) = 0;
    virtual bool is_visited(id_t) = 0;
    virtual void set_visited(id_t) = 0;
    virtual void prefetch_visited_list(id_t) = 0;
    virtual void releaseVisitedList() = 0;
};

class Search_PriorityQueue : public SearchList
{
    struct CompareByFirst {
        constexpr bool operator()(std::pair<float, id_t> const& a,
            std::pair<float, id_t> const& b) const noexcept {
            return a.first < b.first;
        }
    };
    
public:
    uint32_t L;
    std::priority_queue<std::pair<float, id_t>, std::vector<std::pair<float, id_t> >, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<float, id_t>, std::vector<std::pair<float, id_t> >, CompareByFirst> candidate_set;
    std::shared_ptr<VisitedListPool> visited_list_pool;
    VisitedList *vl;
    vl_type *visited_array;
    vl_type visited_array_tag;

    Search_PriorityQueue(uint32_t L, std::shared_ptr<VisitedListPool> visited_list_pool_) {
        this->visited_list_pool = visited_list_pool_;
        this->L = L;
        vl = visited_list_pool_->getFreeVisitedList();
        visited_array = vl->mass;
        visited_array_tag = vl->curV;
    }

    virtual float get_dist_bound() {
        return top_candidates.top().first;
    }

    virtual void push(id_t candidate_id, float dist) {
        bool flag_consider_candidate;
        flag_consider_candidate = top_candidates.size() < L || get_dist_bound() > dist;

        if (flag_consider_candidate) {
            candidate_set.emplace(-dist, candidate_id);
            top_candidates.emplace(dist, candidate_id);
            bool flag_remove_extra = false;
            flag_remove_extra = top_candidates.size() > L;
            while (flag_remove_extra) {
                id_t id = top_candidates.top().second;
                top_candidates.pop();
                flag_remove_extra = top_candidates.size() > L;
            }
        }
    }

    virtual std::pair<float, id_t> get_next_id() {
        auto current_node_pair = candidate_set.top();
        candidate_set.pop();
        return current_node_pair;
    }

    virtual bool is_empty() {
        return candidate_set.empty();
    }


    virtual std::vector<std::pair<float, id_t> > get_result(size_t k) {
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        std::vector<std::pair<float, id_t> > res;
        while(top_candidates.size()) {
            res.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    virtual bool is_visited(id_t id) {
        return visited_array[id] == visited_array_tag;
    }

    virtual void set_visited(id_t id) {
        visited_array[id] = visited_array_tag;
    }

    virtual void prefetch_visited_list(id_t id) {
        _mm_prefetch((char *)(visited_array + id), _MM_HINT_T0);
    }

    virtual void releaseVisitedList() {
        visited_list_pool->releaseVisitedList(vl);
    };
};


class Search_Array : public SearchList {
    struct Candidate {
        id_t id;
        float dist;
        bool is_checked;

        Candidate() = default;
        Candidate(float dist_, id_t id_, bool is_checked_) : id(id_), dist(dist_), is_checked(is_checked_) {}
        bool operator<(const Candidate &b) const
        {
            if (this->dist != b.dist) {
                return this->dist < b.dist;
            } else {
                return this->id < b.id;
            }
        }
    };

public:
    size_t L;
    size_t L_size;
    size_t L_start_idx;
    std::vector<Candidate> L_set;
    std::shared_ptr<VisitedListPool> visited_list_pool;
    VisitedList *vl;
    vl_type *visited_array;
    vl_type visited_array_tag;

    Search_Array(uint32_t L, std::shared_ptr<VisitedListPool> visited_list_pool_){
        this->L = L;
        this->visited_list_pool = visited_list_pool_;
        vl = visited_list_pool_->getFreeVisitedList();
        visited_array = vl->mass;
        visited_array_tag = vl->curV;
        L_set.resize(L);
        L_size = 0;
        L_start_idx = 0;
    }

    void print_queue() {
        std::cout<<"---------------------------------------\n";
        std::cout <<"L = " << L << "\n";
        std::cout<<"sz = "<<L_size<<"  start_idx = "<<L_start_idx<<"\n";
        for(int i = 0; i < L_size; ++i){
            std::cout<<L_set[i].dist<<" "<<L_set[i].is_checked<<" | ";
        }
        std::cout<<"\n";
        std::cout<<"---------------------------------------\n";
    }

    float get_dist_bound() {
        if(L_size == L) {
            return L_set[L - 1].dist;
        }
        return 1ll << 30;
    }

    size_t __push(id_t id, float dist) {
        size_t& sz = L_size;
        size_t end = sz;
        if (sz == 0) {
            L_set[sz] = Candidate{dist, id, false}; 
            sz++;
            return 0; 
        }
        const auto it_loc = std::lower_bound(L_set.begin(), L_set.begin() + end, Candidate{dist, id, false});
        size_t insert_loc = it_loc - L_set.begin(); // start from 0
        if(insert_loc != end) {
            if (id == it_loc->id) {
                // Duplicate, ignore
                return L;
            }
            if (sz >= L) { // Queue is full, will memmove at below
                --sz;
                --end;
            }
        } else{ // insert at the end
            if (sz < L) { // Queue is not full
                // Insert at the end
                L_set[insert_loc] = Candidate{dist, id, false}; 
                ++sz;
                return sz - 1;
            } else { // Queue is full
                return L;
            }
        }

        // now we insert the cand
        // first memmove (rshift)
        memmove(L_set.data() + insert_loc + 1, L_set.data() + insert_loc, (end - insert_loc) * sizeof(Candidate));
        L_set[insert_loc] = Candidate{dist, id, false};
        ++sz;
        return insert_loc;
    }

    void push(id_t id, float dist) {
        if(dist > get_dist_bound()) {
            return;
        }
        size_t insert_idx = __push(id, dist);
        if(L_start_idx > insert_idx) {
            L_start_idx = insert_idx;
        }
    }

    // need to call is_empty() first
    virtual std::pair<float, id_t> get_next_id() {
        L_set[L_start_idx].is_checked = true;
        return {L_set[L_start_idx].dist, L_set[L_start_idx].id};
    }

    virtual bool is_empty() {
        while(L_start_idx < L_size && L_set[L_start_idx].is_checked) {
            ++L_start_idx;
        }
        return L_start_idx >= L_size;
    }

    virtual std::vector<std::pair<float, id_t> > get_result(size_t k) {
        std::vector<std::pair<float, id_t> > res;
        res.resize(k);
        for(int i = 0; i < k; ++i) {
            res[i] = {L_set[i].dist, L_set[i].id};
        }
        return res;
    }


    virtual bool is_visited(id_t id) {
        return visited_array[id] == visited_array_tag;
    }

    virtual void set_visited(id_t id) {
        visited_array[id] = visited_array_tag;
    }

    virtual void prefetch_visited_list(id_t id) {
        _mm_prefetch((char *)(visited_array + id), _MM_HINT_T0);
    }

    virtual void releaseVisitedList() {
        visited_list_pool->releaseVisitedList(vl);
    };
};


class Search_QuadHeap : public SearchList {

    template<typename T> 
    struct QuadHeap {
        std::vector<T> data;
        size_t sz;
        static constexpr size_t HEAP_ARY = 4;
        QuadHeap(size_t ef) {
            sz = 0;
            data.reserve(ef);
        }

        void push(const T& item) {
            if (sz < data.size()) {
                data[sz] = item;
            } else {
                data.push_back(item);
            }
            heapify_up(sz);
            ++sz;
        }

        void pop() {
            --sz;
            if (sz > 0) {
                data[0] = data[sz];
                heapify_down(0);
            } 
        }

        T top() {
            return data[0];
        }

        size_t size() {
            return sz;
        }

        bool empty() {
            return sz == 0;
        }

    private: 
        void heapify_up(size_t idx) {
            while (idx > 0) {
                size_t parent = (idx - 1) / HEAP_ARY;
                if (data[parent] < data[idx]) {
                    std::swap(data[parent], data[idx]);
                    idx = parent;
                } else {
                    break;
                }
            }
        }

        void heapify_down(size_t idx) {
            while (true) {
                size_t first_child = HEAP_ARY * idx + 1;
                if (first_child >= sz) {break; }

                size_t largest = idx;

                if (data[largest] < data[first_child]) {
                    largest = first_child;
                }
                if (first_child + 1 < sz && data[largest] < data[first_child + 1]) {
                    largest = first_child + 1;
                }
                if (first_child + 2 < sz && data[largest] < data[first_child + 2]) {
                    largest = first_child + 2;
                }
                if (first_child + 3 < sz && data[largest] < data[first_child + 3]) {
                    largest = first_child + 3;
                }

                if (largest != idx) {
                    std::swap(data[idx], data[largest]);
                    idx = largest;
                } else {
                    break;
                }
            }
        }
    };
public:
    uint32_t L;
    QuadHeap<std::pair<float, id_t> > top_candidates;
    QuadHeap<std::pair<float, id_t> > candidate_set;
    std::shared_ptr<VisitedListPool> visited_list_pool;
    VisitedList *vl;
    vl_type *visited_array;
    vl_type visited_array_tag;

    Search_QuadHeap(uint32_t L, std::shared_ptr<VisitedListPool> visited_list_pool_)
    : top_candidates(L), candidate_set(L) {
        this->visited_list_pool = visited_list_pool_;
        this->L = L;
        vl = visited_list_pool_->getFreeVisitedList();
        visited_array = vl->mass;
        visited_array_tag = vl->curV;
    }

    virtual float get_dist_bound() {
        return top_candidates.top().first;
    }

    virtual void push(id_t candidate_id, float dist) {
        bool flag_consider_candidate;
        flag_consider_candidate = top_candidates.size() < L || get_dist_bound() > dist;

        if (flag_consider_candidate) {
            candidate_set.push({-dist, candidate_id});
            top_candidates.push({dist, candidate_id});
            bool flag_remove_extra = false;
            flag_remove_extra = top_candidates.size() > L;
            while (flag_remove_extra) {
                id_t id = top_candidates.top().second;
                top_candidates.pop();
                flag_remove_extra = top_candidates.size() > L;
            }
        }
    }

    virtual std::pair<float, id_t> get_next_id() {
        auto current_node_pair = candidate_set.top();
        candidate_set.pop();
        return current_node_pair;
    }

    virtual bool is_empty() {
        return candidate_set.empty();
    }


    virtual std::vector<std::pair<float, id_t> > get_result(size_t k) {
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        std::vector<std::pair<float, id_t> > res;
        while(top_candidates.size()) {
            res.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    virtual bool is_visited(id_t id) {
        return visited_array[id] == visited_array_tag;
    }

    virtual void set_visited(id_t id) {
        visited_array[id] = visited_array_tag;
    }

    virtual void prefetch_visited_list(id_t id) {
        _mm_prefetch((char *)(visited_array + id), _MM_HINT_T0);
    }

    virtual void releaseVisitedList() {
        visited_list_pool->releaseVisitedList(vl);
    };
};
}