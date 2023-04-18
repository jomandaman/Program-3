cmake_minimum_required(VERSION 3.15)

set (CMAKE_CXX_STANDARD 17)

project(program3)

set(SOURCE Program3.cpp)

INCLUDE_DIRECTORIES(/usr/local/include/opencv4)

LINK_DIRECTORIES(/usr/local/lib)

add_executable(${PROJECT_NAME} ${SOURCE})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} libopencv_core.dylib libopencv_highgui.dylib libopencv_imgcodecs.dylib libopencv_imgproc.dylib libopencv_features2d.dylib libopencv_xfeatures2d.dylib)
