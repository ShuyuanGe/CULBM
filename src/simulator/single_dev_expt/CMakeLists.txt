file(GLOB single_dev_expt_src ./*.cpp ./*.cu)

function(add_single_dev_expt_library target_name)
    add_library(${target_name} STATIC ${single_dev_expt_src})

    target_include_directories(${target_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(${target_name} PUBLIC lbm_core blocking_core)

    target_compile_options(
        ${target_name}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -lineinfo>
    )

    # Enable CUDA separable compilation (-rdc=true), so __global__ template instantiations
    # that are declared in .cuh files can be defined in .cu files without triggering warning #20280-D.
    set_target_properties(
        ${target_name}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

add_single_dev_expt_library(single_dev_expt)
add_single_dev_expt_library(single_dev_expt_no_persist_no_hint)
add_single_dev_expt_library(single_dev_expt_no_persist_hint_only)

target_compile_definitions(
    single_dev_expt_no_persist_no_hint
    PRIVATE
    CULBM_EXPT_ENABLE_PERSISTENT_L2=0
    CULBM_EXPT_CACHE_HINT_POLICY=0
)

target_compile_definitions(
    single_dev_expt_no_persist_hint_only
    PRIVATE
    CULBM_EXPT_ENABLE_PERSISTENT_L2=0
    CULBM_EXPT_CACHE_HINT_POLICY=1
)
