file(GLOB single_dev_expt_src ./*.cpp ./*.cu)

add_library(single_dev_expt STATIC ${single_dev_expt_src})

target_include_directories(single_dev_expt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(single_dev_expt PUBLIC lbm_core blocking_core)

target_compile_options(
    single_dev_expt 
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -lineinfo>
)