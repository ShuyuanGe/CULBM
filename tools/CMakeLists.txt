add_executable(device_prop device_prop.cpp)
target_link_libraries(device_prop PRIVATE culbm::gpu)
