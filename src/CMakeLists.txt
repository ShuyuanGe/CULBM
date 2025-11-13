add_subdirectory(third_party)
add_subdirectory(basic)
add_subdirectory(blocking_core)
add_subdirectory(lbm_core)
add_subdirectory(simulator)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE simulator_single_dev)