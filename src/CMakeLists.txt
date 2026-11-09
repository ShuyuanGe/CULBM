add_subdirectory(third_party)
add_subdirectory(util)
add_subdirectory(gpu)
add_subdirectory(core)
add_subdirectory(sim)

add_executable(multi_dev_main multi_dev_main.cpp)
target_link_libraries(multi_dev_main PRIVATE culbm::sim_multi)

add_executable(single_dev_main single_dev_main.cpp)
target_link_libraries(single_dev_main PRIVATE culbm::sim_single)
