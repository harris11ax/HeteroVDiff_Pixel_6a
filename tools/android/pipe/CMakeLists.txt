cmake_minimum_required(VERSION 3.22)
project(pixel6a_pipe CXX)

set(CMAKE_CXX_STANDARD 17)

# phase3_pipeline/handoff.h is the shared ring definition — P3.2/P3.3 reuse it.
add_executable(handoff_selftest handoff_selftest.cpp)
target_include_directories(handoff_selftest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../phase3_pipeline")

# P3.2 — the GPU stage over the real INT8 artifact (route B: mmap + clImportMemoryARM).
add_executable(proc_a proc_a.cpp)
target_include_directories(proc_a PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../phase3_pipeline")
target_link_libraries(proc_a dl log)
