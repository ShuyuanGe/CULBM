add_library(CLI11 STATIC CLI11.cpp)

target_include_directories(CLI11 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_precompile_headers(CLI11 PUBLIC CLI11.hpp)