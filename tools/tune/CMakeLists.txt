add_library(culbm_tune_measurements STATIC lambda.cu gamma.cu)
target_link_libraries(culbm_tune_measurements PUBLIC culbm::gpu)

add_executable(measure_lambda lambda.cpp)
target_link_libraries(measure_lambda PRIVATE culbm_tune_measurements)

add_executable(measure_gamma gamma.cpp)
target_link_libraries(measure_gamma PRIVATE culbm_tune_measurements)

add_executable(culbm-tune main.cpp selector.cpp)
target_link_libraries(culbm-tune PRIVATE culbm_tune_measurements CLI11)
