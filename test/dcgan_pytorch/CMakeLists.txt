cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(dcgan)
add_executable(dcgan dcgan.cpp)

# LibTorch
set(CMAKE_INCLUDE_DIRECTORY ../../include) 
find_package(Torch REQUIRED PATHS "${CMAKE_INCLUDE_DIRECTORY}/libtorch_CUDA12_1/share/cmake/Torch")
target_link_libraries(dcgan "${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 17)