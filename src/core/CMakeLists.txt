add_library(core INTERFACE)
add_library(culbm::core ALIAS core)

target_link_libraries(core INTERFACE CCCL::CCCL)
target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(core INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
