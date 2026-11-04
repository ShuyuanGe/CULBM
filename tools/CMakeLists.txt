add_executable(device_prop device_prop.cpp)
target_compile_features(device_prop PRIVATE cxx_std_23)
target_link_libraries(device_prop PRIVATE CUDA::cudart basic)
