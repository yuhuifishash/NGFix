#pragma once

#include <vector>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <mutex>

#define LOG

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    #ifdef LOG
        std::cout<<"load data "<<data_path<<"\n";
        std::cout<<"dimension: "<<d<<"  number: "<<n<<"\n";
    #endif
    return data;
}

// used for convert sift.u8bin to sift.fbin
float* u8_f32(uint8_t* data, size_t n)
{
    float* f32_data = new float[n];
    for(int i = 0; i < n; ++i){
        f32_data[i] = data[i];
    }
    return f32_data;
}

auto getVectorsHead(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);

    #ifdef LOG
        std::cout<<"load data head "<<data_path<<"\n";
        std::cout<<"dimension: "<<d<<"  number: "<<n<<"\n";
    #endif

    return fin;
}


static std::mutex loader_lock;
template<typename T>
T* getNextVector(std::ifstream& fin, size_t d)
{
    T* data = new T[d];
    int sz = sizeof(T);
    loader_lock.lock();
    fin.read((char*)data, d*sz);
    loader_lock.unlock();
    
    return data;
}