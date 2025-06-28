cmake_minimum_required(VERSION 3.22)
project(DeepSORT)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -Wno-deprecated-declarations")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(modules/tensorrt-cpp-api)

# Build the YOLOv8 library
include_directories(include)
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC modules/tensorrt-cpp-api/src)

# Build the ReID library
add_library(ReID_TRT SHARED src/reid_engine.cpp)
target_link_libraries(ReID_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(ReID_TRT PUBLIC modules/tensorrt-cpp-api/src)

# Build the tracking library
add_library(tracking SHARED
    src/detection.cpp
    src/kalman_filter.cpp
    src/matching.cpp
    src/track.cpp
    src/tracker.cpp
    src/munkres.cpp
)

target_include_directories(tracking PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tracking PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

# Build DeepSORT example
add_executable(deep_sort_example examples/deep_sort_example.cpp)
target_link_libraries(deep_sort_example 
    tracking
    YoloV8_TRT
    ReID_TRT
)