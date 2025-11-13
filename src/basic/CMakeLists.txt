file(GLOB basic_src ./*.cpp ./*.cu)

add_library(basic STATIC ${basic_src})

target_include_directories(basic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(basic PUBLIC CLI11)