add_library(util INTERFACE)
add_library(culbm::util ALIAS util)
target_include_directories(util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(Threads REQUIRED)
add_library(workers STATIC logger.cpp workers.cpp)
add_library(culbm::workers ALIAS workers)
target_link_libraries(workers PUBLIC culbm::util Threads::Threads)
