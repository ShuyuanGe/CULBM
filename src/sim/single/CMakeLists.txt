add_library(sim_single STATIC config.cpp simulator.cpp state.cpp kernel.cu)
add_library(culbm::sim_single ALIAS sim_single)

target_include_directories(sim_single PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(sim_single PUBLIC ${CULBM_CUDART} culbm::core PRIVATE CLI11 culbm::gpu culbm::physical)

target_compile_options(
    sim_single
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -lineinfo>
)

set_target_properties(sim_single PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(culbm-single main.cpp)
target_link_libraries(culbm-single PRIVATE culbm::sim_single)

if(CULBM_STANDALONE)
    culbm_remove_rpath(culbm-single)
endif()
