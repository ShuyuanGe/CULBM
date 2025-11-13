file(GLOB simulator_single_dev_src ./*.cpp ./*.cu)

add_library(simulator_single_dev STATIC ${simulator_single_dev_src})

target_include_directories(simulator_single_dev PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(simulator_single_dev PUBLIC lbm_core blocking_core)