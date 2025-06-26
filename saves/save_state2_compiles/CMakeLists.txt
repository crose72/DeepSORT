cmake_minimum_required(VERSION 3.10)
project(tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

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