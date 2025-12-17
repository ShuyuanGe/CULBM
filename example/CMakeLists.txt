add_executable(device_prop device_prop.cu)
target_link_libraries(device_prop PRIVATE basic)

add_executable(stream_mixblk_compress stream_mixblk_compress.cu)
target_link_libraries(stream_mixblk_compress PRIVATE basic)
target_compile_options(stream_mixblk_compress PRIVATE --expt-relaxed-constexpr)

add_executable(inplace_stream_check inplace_stream_check.cu)
target_link_libraries(inplace_stream_check PRIVATE basic)
target_compile_options(inplace_stream_check PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)