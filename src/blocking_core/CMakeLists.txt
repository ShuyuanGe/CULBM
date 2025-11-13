add_library(blocking_core INTERFACE)

target_include_directories(blocking_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(blocking_core INTERFACE basic)