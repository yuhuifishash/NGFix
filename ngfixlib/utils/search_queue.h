#pragma once
#include "../ngfixlib/graph/node.h"
#include <queue>
#include <algorithm>

namespace ngfixlib {

class SearchQueue
{
public:
    virtual float get_dist_bound() = 0;
    virtual void push(id_t id, float dist) = 0;
    virtual std::pair<float, id_t> get_next_id() = 0; // also pop
    virtual bool is_empty() = 0;
    virtual std::vector<std::pair<float, id_t> > get_result(size_t k) = 0;
};


struct CompareByFirst {
    constexpr bool operator()(std::pair<float, id_t> const& a,
        std::pair<float, id_t> const& b) const noexcept {
        return a.first < b.first;
    }
};

class Search_PriorityQueue : public SearchQueue
{
public:
    uint32_t L;
    std::priority_queue<std::pair<float, id_t>, std::vector<std::pair<float, id_t> >, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<float, id_t>, std::vector<std::pair<float, id_t> >, CompareByFirst> candidate_set;
    Search_PriorityQueue(uint32_t L) {
        this->L = L;
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
};

}