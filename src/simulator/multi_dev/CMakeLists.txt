file(GLOB simulator_multi_dev_src ./*.cpp ./*.cu)

add_library(simulator_multi_dev STATIC ${simulator_multi_dev_src})

target_include_directories(simulator_multi_dev PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(simulator_multi_dev PUBLIC CUDA::cudart lbm_core)
