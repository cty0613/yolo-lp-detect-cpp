cmake_minimum_required(VERSION 3.10)
project(yolov5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ncnn 패키지 찾기
set(ncnn_DIR "/home/cty0613/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

# OpenCV 찾기 (pkg-config 활용)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/usr/include/tesseract)

# Threads
find_package(Threads REQUIRED)

add_executable(yolov5 yolo.cpp main.cpp plate.cpp)

# 라이브러리 링크
target_link_libraries(yolov5
    ${OpenCV_LIBRARIES}  # OpenCV 라이브러리
    ncnn                # ncnn 라이브러리
    /usr/lib/aarch64-linux-gnu/libtesseract.so
    Threads::Threads    # CMake Threads 방식
    rt
)
