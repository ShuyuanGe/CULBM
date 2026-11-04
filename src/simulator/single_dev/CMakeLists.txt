add_library(single_dev STATIC config.cpp runner.cpp simulator.cpp state.cpp strategy.cpp kernel.cu)

target_include_directories(single_dev PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(single_dev PUBLIC CUDA::cudart core PRIVATE CUDA::cuda_driver)

target_compile_options(
    single_dev
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -lineinfo>
)

# Template kernels are defined in .cu files and dispatched by the host.
set_target_properties(
    single_dev
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
