cmake_minimum_required(VERSION 3.22)
project(pixel6a_probe CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(probe probe.cpp)

# android    -> AHardwareBuffer_* (libandroid)
# neuralnetworks -> NNAPI (deprecated in Android 15 but still present/served on Pixel)
# dl, log    -> dlsym/libOpenCL.so and __android_log
target_link_libraries(probe android neuralnetworks dl log)

# No OpenCL headers on purpose: the probe resolves libOpenCL.so with dlopen()
# and declares the handful of CL types it needs, so nothing is required at
# link time and the same binary runs even if a vendor ships no CL stub.
