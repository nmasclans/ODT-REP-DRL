cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(end_to_end)
add_executable(end_to_end main.cc)

# LibTorch
set(CMAKE_INCLUDE_DIRECTORY ../../include) 
find_package(Torch REQUIRED PATHS "${CMAKE_INCLUDE_DIRECTORY}/libtorch_CUDA12_1/share/cmake/Torch")
target_link_libraries(end_to_end "${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set_property(TARGET end_to_end PROPERTY CXX_STANDARD 17)

#####################################################################

target_include_directories(end_to_end PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(end_to_end
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dqn.cc            ${CMAKE_CURRENT_SOURCE_DIR}/dqn.h
)