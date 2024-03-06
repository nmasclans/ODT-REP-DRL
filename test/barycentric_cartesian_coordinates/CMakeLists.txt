cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

# Define the executable with its initial source files
add_executable(test.x main.cc dv_reynolds_stress.cc)  # Assuming dv_reynolds_stress.cc is the implementation file

# Set the C++ standard for the target
set_property(TARGET test.x PROPERTY CXX_STANDARD 17)

target_include_directories(test.x PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})