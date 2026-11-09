add_library(gpu STATIC error.cpp fill.cu)
add_library(culbm::gpu ALIAS gpu)

target_include_directories(gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(gpu PUBLIC CUDA::cudart CUDA::cuda_driver CCCL::CCCL)
# glibc < 2.34: keep libpthread for weak references under --as-needed.
target_link_libraries(gpu PUBLIC "LINKER:--push-state,--no-as-needed" pthread "LINKER:--pop-state")
set_target_properties(gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
