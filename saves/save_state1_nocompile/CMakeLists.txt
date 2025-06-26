cmake_minimum_required(VERSION 3.22)
project(DeepSORT)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# CMake Options
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(modules/tensorrt-cpp-api)

# Build the YoloV8 library
include_directories(include)
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC modules/tensorrt-cpp-api/src)

# Build and link the examples for YoloV8
add_executable(detect_object_image examples/object_detection_image.cpp)
target_link_libraries(detect_object_image YoloV8_TRT)

add_executable(detect_object_video examples/object_detection_video_stream.cpp)
target_link_libraries(detect_object_video YoloV8_TRT)

add_executable(detect_object_csi_jetson examples/object_detection_csi_jetson.cpp)
target_link_libraries(detect_object_csi_jetson YoloV8_TRT)

# Build DeepSORT library
file(GLOB DeepSORT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(DeepSORT SHARED ${DeepSORT_SRC})

target_include_directories(DeepSORT PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    modules/tensorrt-cpp-api/src
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(DeepSORT PUBLIC
    tensorrt_cpp_api
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Build DeepSORT example
add_executable(deep_sort_example examples/deep_sort_example.cpp)
target_link_libraries(deep_sort_example DeepSORT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
