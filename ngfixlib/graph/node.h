#pragma once

#include <vector>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <atomic>
#include <mutex>
#include <cstring>
#include <condition_variable>

namespace ngfixlib {

typedef unsigned int id_t;
const int CAPACITY_INC = 4;
const int EH_INF = std::numeric_limits<uint16_t>::max();
const float INF_RATIO = 0.2;


uint8_t GET_CAPACITY(uint8_t* array) {
    return *(array);
}

uint8_t GET_SZ(uint8_t* array) {
    return *(array + 1);
}

uint8_t GET_NGFIX_CAPACITY(uint8_t* array) {
    return *(array + 2);
}

uint8_t GET_NGFIX_SZ(uint8_t* array) {
    return *(array + 3);
}

void SET_CAPACITY(uint8_t* array, uint8_t val) {
    *array = val;
}

void SET_SZ(uint8_t* array, uint8_t val) {
    *(array + 1) = val;
}

void SET_NGFIX_CAPACITY(uint8_t* array, uint8_t val) {
    *(array + 2) = val;
}

void SET_NGFIX_SZ(uint8_t* array, uint8_t val) {
    *(array + 3) = val;
}

struct node
{
    /*
        edge layout [***>>>>>>>>>>>>>>>>>>, >>>>>>>>>>>*** ]   (*:empty,  >:real edges)
                          ngfix_edges       base_graph_edges
        
        sz = number of ">"
        capacity = number of ">" + number of "*"
    */


    // neighbros => [{capacity, sz, delete_flag, ref_count}, edges]
    id_t* neighbors = nullptr;
    uint16_t* ehs = nullptr;

    node() {
        neighbors = new id_t[CAPACITY_INC + 1];
        neighbors[0] = 0;
        SET_CAPACITY((uint8_t*)neighbors, CAPACITY_INC);
    }

    id_t* get_neighbors() {
        return neighbors;
    }

    void delete_node() {
        delete []neighbors;
        delete []ehs;
        neighbors = new id_t[CAPACITY_INC + 1];
        neighbors[0] = 0;
        SET_CAPACITY((uint8_t*)neighbors, CAPACITY_INC);
    }

    void add_base_graph_neighbors(id_t v) {
        uint8_t sz = GET_SZ((uint8_t*)neighbors);
        uint8_t capacity = GET_CAPACITY((uint8_t*)neighbors);
        uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        uint8_t base_sz = sz - GET_NGFIX_SZ((uint8_t*)neighbors);
        uint8_t base_capacity = capacity - ngfix_capacity;

        sz += 1;
        base_sz += 1;

        if(base_sz > base_capacity) {
            capacity += CAPACITY_INC;

            auto n = new id_t[capacity + 1];
            memcpy(n, neighbors, sizeof(id_t)*(capacity - CAPACITY_INC + 1));
            n[ngfix_capacity + base_sz] = v;
            SET_SZ((uint8_t*)n, sz);
            SET_CAPACITY((uint8_t*)n, capacity);
            
            auto tmp = neighbors;
            neighbors = n;
            delete []tmp;
        } else {
            neighbors[ngfix_capacity + base_sz] = v;
            SET_SZ((uint8_t*)neighbors, sz);
        }
        
    }

    // new_neighbors.size() <= 2M
    void replace_base_graph_neighbors(std::vector<std::pair<float, id_t> >& new_neighbors) {
        uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
        uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        uint8_t base_sz = new_neighbors.size();
        uint8_t sz = base_sz + ngfix_sz;
        uint8_t capacity = ((base_sz + CAPACITY_INC - 1) / CAPACITY_INC) * CAPACITY_INC + ngfix_capacity;

        auto n = new id_t[capacity + 1];
        memcpy(n, neighbors, sizeof(id_t)*(ngfix_capacity + 1));
        for(int i = 0; i < new_neighbors.size(); ++i) {
            n[ngfix_capacity + i + 1] = new_neighbors[i].second;
        }
        SET_SZ((uint8_t*)n, sz);
        SET_CAPACITY((uint8_t*)n, capacity);

        auto tmp = neighbors;
        neighbors = n;
        delete []tmp;
    }

    // used for partial rebuilding, eh is set to 0
    void replace_ngfix_neighbors(std::vector<id_t>& new_neighbors) {
        uint8_t sz = GET_SZ((uint8_t*)neighbors);
        uint8_t capacity = GET_CAPACITY((uint8_t*)neighbors);
        uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
        uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        uint8_t base_sz = sz - ngfix_sz;
        uint8_t base_capacity = capacity - ngfix_capacity;

        uint8_t new_ngfix_sz = new_neighbors.size();
        uint8_t new_ngfix_capacity = ((new_ngfix_sz + CAPACITY_INC - 1) / CAPACITY_INC) * CAPACITY_INC;
        uint8_t new_sz = base_sz + new_ngfix_sz;
        uint8_t new_capacity = base_capacity + new_ngfix_capacity;

        auto n = new id_t[new_capacity + 1];
        SET_CAPACITY((uint8_t*)n, new_capacity);
        SET_SZ((uint8_t*)n, new_sz);
        SET_NGFIX_CAPACITY((uint8_t*)n, new_ngfix_capacity);
        SET_NGFIX_SZ((uint8_t*)n, new_ngfix_sz);

        if (base_sz > 0) {
            memcpy(n + new_ngfix_capacity + 1, neighbors + ngfix_capacity + 1, sizeof(id_t) * base_sz);
        }
        for (int i = 0; i < new_ngfix_sz; ++i) {
            n[new_ngfix_capacity - i] = new_neighbors[i];
        }

        if (new_ngfix_capacity > 0) {
            auto n_ehs = new uint16_t[new_ngfix_capacity];

            for (int i = 0; i < new_ngfix_capacity; ++i) {
                n_ehs[i] = 0;
            }
            if (ehs != nullptr) {
                delete [] ehs;
            }
            ehs = n_ehs;
        } else {
            if (ehs != nullptr) {
                delete [] ehs;
                ehs = nullptr;
            }
        }
        
        auto tmp = neighbors;
        neighbors = n;
        delete []tmp;
    }

    void add_ngfix_neighbors(id_t v, uint16_t eh, size_t MEX) {
        uint8_t sz = GET_SZ((uint8_t*)neighbors);
        uint8_t capacity = GET_CAPACITY((uint8_t*)neighbors);
        uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
        uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        // printf("%d %d %d %d %d\n", sz, capacity, ngfix_sz, ngfix_capacity, MEX);
        if(ngfix_sz == MEX) { // prune edges
            // assuming MEX % CAPACITY_INC = 0
            size_t inf_cnt = 0;
            size_t min_idx = 0;
            size_t min_eh = EH_INF;
            for(int i = 0; i < MEX; ++i) {
                if(ehs[i] < min_eh) {
                    min_eh = ehs[i];
                    min_idx = i;
                }
                if(ehs[i] == EH_INF) {
                    ++inf_cnt;
                }
            }
            // Too many long edges (i.e., edges with inf EH) will improve performance when L >= MAX_S (high recall),
            // but will decrease the search performance when L is low (e.g. moderate recall or low recall).
            // Therefore, we limit the number of edges with inf EH.
            if(eh == EH_INF && inf_cnt >= (float)MEX * INF_RATIO) {return;}
            if(eh < min_eh) {return;}
            ehs[min_idx] = eh;
            neighbors[min_idx + 1] = v;

        } else{
            ngfix_sz += 1;
            sz += 1;
            if(ngfix_sz > ngfix_capacity) {
                capacity += CAPACITY_INC;
                ngfix_capacity += CAPACITY_INC;

                auto n = new id_t[capacity + 1];
                memcpy(n + 1 + CAPACITY_INC, neighbors + 1, sizeof(id_t)*(capacity - CAPACITY_INC));
                n[ngfix_capacity - ngfix_sz + 1] = v;

                SET_SZ((uint8_t*)n, sz);
                SET_CAPACITY((uint8_t*)n, capacity);
                SET_NGFIX_SZ((uint8_t*)n, ngfix_sz);
                SET_NGFIX_CAPACITY((uint8_t*)n, ngfix_capacity);
                
                auto tmp = neighbors;
                neighbors = n;
                delete []tmp;

                auto n_ehs = new uint16_t[ngfix_capacity];
                memcpy(n_ehs + CAPACITY_INC, ehs, sizeof(uint16_t)*(ngfix_capacity - CAPACITY_INC));
                n_ehs[ngfix_capacity - ngfix_sz] = eh;

                auto tmp2 = ehs;
                ehs = n_ehs;
                if(tmp2 != nullptr) {
                    delete []tmp2;
                }

            } else {
                ehs[ngfix_capacity - ngfix_sz] = eh;
                neighbors[ngfix_capacity - ngfix_sz + 1] = v;
                SET_SZ((uint8_t*)neighbors, sz);
                SET_NGFIX_SZ((uint8_t*)neighbors, ngfix_sz);
            }
        }

    }

    void StoreIndex(std::ofstream& s) {
        uint8_t capacity = GET_CAPACITY((uint8_t*)neighbors);
        uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        s.write((char*)neighbors, sizeof(id_t)*(capacity + 1));
        if(ngfix_capacity > 0) {
            s.write((char*)ehs, sizeof(uint16_t)*ngfix_capacity);
        }
    }

    void LoadIndex(std::ifstream& s) {
        id_t meta;
        s.read((char*)(&meta), sizeof(id_t));
        
        uint8_t* meta_bytes = (uint8_t*)(&meta);
        uint8_t capacity = meta_bytes[0];
        uint8_t sz = meta_bytes[1];
        uint8_t ngfix_capacity = meta_bytes[2];
        uint8_t ngfix_sz = meta_bytes[3];
        
        delete []neighbors;
        neighbors = new id_t[capacity + 1];
        neighbors[0] = meta;
        
        if (capacity > 0) {
            s.read((char*)(neighbors + 1), sizeof(id_t)*capacity);
        }
        
        if (ngfix_capacity > 0) {
            if(ehs != nullptr) {delete []ehs;}
            ehs = new uint16_t[ngfix_capacity];
            s.read((char*)(ehs), sizeof(uint16_t)*ngfix_capacity);
        }
    }
};
}