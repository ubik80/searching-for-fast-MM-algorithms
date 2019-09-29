cmake_minimum_required(VERSION 2.8.12)
project(backprop)

set(CMAKE_CXX_FLAGS "-Wall -O3 -march=native")

add_subdirectory(pybind11)
pybind11_add_module(backprop backprop.cpp)
