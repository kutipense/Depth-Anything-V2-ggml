add_compile_options(-O3)

set(TEST_TARGET dptv2)
add_executable(${TEST_TARGET} dptv2.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
