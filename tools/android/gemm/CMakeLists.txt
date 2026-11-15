cmake_minimum_required(VERSION 3.22)
project(pixel6a_gemm CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(gemm_bench gemm_bench.cpp)
target_link_libraries(gemm_bench dl log)

# The kernel source (kernels/gemm8.cl) is pushed to the device separately and read
# at run time, so kernel edits never require a rebuild of this binary.
