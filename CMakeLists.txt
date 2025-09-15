cmake_minimum_required(VERSION 3.0...3.26)

project(NGFix LANGUAGES CXX)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/rabitqlib)
include_directories(${PROJECT_SOURCE_DIR}/test)



SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++20 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

add_subdirectory(test)
