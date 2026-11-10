add_library(physical SHARED resource.cpp capture.cpp)
add_library(culbm::physical ALIAS physical)

target_include_directories(physical PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(physical PUBLIC CUDA::cudart CCCL::CCCL)
target_link_libraries(physical PRIVATE culbm::gpu)
