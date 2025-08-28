cmake_minimum_required(VERSION 3.0...3.26)

project(EXlib LANGUAGES CXX)

include_directories(${PROJECT_SOURCE_DIR}/ngfixlib/graph)
include_directories(${PROJECT_SOURCE_DIR}/ngfixlib/metric)
include_directories(${PROJECT_SOURCE_DIR}/ngfixlib/utils)


SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++20 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

add_executable(build_hnsw_ngfix test/build_hnsw_ngfix.cc)
add_executable(search_hnsw_ngfix test/search_hnsw_ngfix.cc)
