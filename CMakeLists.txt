cmake_minimum_required(VERSION 4.0.1)

project(cuLBM LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(CCCL 3.2 REQUIRED CONFIG)

option(CULBM_STANDALONE "Build standalone executables for Linux" OFF)
set(CULBM_CUDART CUDA::cudart)
if(CULBM_STANDALONE)
    set(CULBM_CUDART CUDA::cudart_static)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    add_link_options(-static-libstdc++ -static-libgcc)

    function(culbm_remove_rpath target)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} "-DFILE=$<TARGET_FILE:${target}>" -P "${PROJECT_SOURCE_DIR}/cmake/remove_rpath.cmake"
            VERBATIM)
    endfunction()
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(tools)
