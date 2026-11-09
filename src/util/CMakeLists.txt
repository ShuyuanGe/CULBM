add_library(util STATIC logger.cpp workers.cpp)
add_library(culbm::util ALIAS util)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(Threads REQUIRED)
target_link_libraries(util PUBLIC Threads::Threads)
