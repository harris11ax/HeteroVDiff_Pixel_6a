cmake_minimum_required(VERSION 3.22)
project(pixel6a_npublock CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(npu_block npu_block.cpp)

# neuralnetworks -> NNAPI (deprecated in AOSP but still served on Pixel)
# The weight blob lives in /data/local/tmp and is opened at run time, so weight
# content never requires a rebuild.
target_link_libraries(npu_block neuralnetworks)
