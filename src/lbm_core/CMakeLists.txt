add_library(lbm_core INTERFACE)

target_link_libraries(lbm_core INTERFACE basic)
target_include_directories(lbm_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(lbm_core INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
