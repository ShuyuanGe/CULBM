add_library(core INTERFACE)

target_link_libraries(core INTERFACE basic)
target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(core INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
