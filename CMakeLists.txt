cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_FLAGS "-g -Wall")

FIND_PACKAGE(OpenGL REQUIRED) #for graphics

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(./)
add_subdirectory(intro)
add_subdirectory(advanced)
