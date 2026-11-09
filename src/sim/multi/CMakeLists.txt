add_library(sim_multi STATIC config.cpp simulator.cpp state.cpp kernel.cu)
add_library(culbm::sim_multi ALIAS sim_multi)

target_include_directories(sim_multi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(sim_multi PUBLIC CUDA::cudart culbm::core PRIVATE culbm::workers CLI11 culbm::gpu)

add_executable(culbm-multi main.cpp)
target_link_libraries(culbm-multi PRIVATE culbm::sim_multi)
