cmake_minimum_required(VERSION 2.8)
project(FindCornerSubpixel)
#set( CMAKE_BUILD_TYPE "Release" )
#set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include "/usr/include/eigen3")

link_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)
ADD_SUBDIRECTORY(src)






