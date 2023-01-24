cmake_minimum_required(VERSION 3.0.0)

cmake_policy(SET CMP0072 NEW)

project(cuda_demos)
set(CMAKE_CXX_FLAGS "-g -Wall")

#Find necessary packages
FIND_PACKAGE(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA QUIET REQUIRED) #Any version OK

#Set CUDA paths
cuda_include_directories(${CUDA_INC} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CUDA_SDK}/lib/)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(./)
add_subdirectory(common)
add_subdirectory(intro)
add_subdirectory(advanced)


