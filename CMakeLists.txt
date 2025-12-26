cmake_minimum_required(VERSION 4.0.1)

project(CULBM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED 20)

add_subdirectory(src)
add_subdirectory(example)