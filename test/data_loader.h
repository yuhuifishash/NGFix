#pragma once

#include <vector>
#include <cstring>
#include <string>

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
        std::cerr<<"load data "<<data_path<<"\n";
        std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";
    #endif
    return data;
}

float* u8_f32(uint8_t* data, size_t n)
{
    float* f32_data = new float[n];
    for(int i = 0; i < n; ++i){
        f32_data[i] = data[i];
    }
    return f32_data;
}