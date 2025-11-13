file(GLOB lbm_core_src ./*.cpp ./*.cu)

add_library(lbm_core STATIC ${lbm_core_src})

target_link_libraries(lbm_core PUBLIC basic)

target_include_directories(lbm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(lbm_core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)