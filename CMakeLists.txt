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
# include_directories(/usr/include/tesseract)

# Threads
find_package(Threads REQUIRED)

# --- TensorFlow Lite 직접 include 및 static lib 지정 ---
set(TFLITE_INCLUDE_DIR "/home/cty0613/tensorflow_src/bazel-tensorflow_src")         # 예: 빌드된 TFLite의 헤더 경로
set(TFLITE_STATIC_LIB "/home/cty0613/tensorflow_src/bazel-bin/tensorflow/lite/libtensorflowlite.so")  # 정적 라이브러리 경로
message(STATUS "Found TF-LITE Includes:" ${TFLITE_INCLUDE_DIR})
message(STATUS "Found TF-LITE Libs:" ${TFLITE_STATIC_LIB})
# 존재 여부 확인
if(NOT EXISTS ${TFLITE_INCLUDE_DIR}/tensorflow/lite/interpreter.h)
    message(FATAL_ERROR "TFLite 헤더가 ${TFLITE_INCLUDE_DIR}에 없음")
endif()

if(NOT EXISTS ${TFLITE_STATIC_LIB})
    message(FATAL_ERROR "TFLite 라이브러리 ${TFLITE_STATIC_LIB}가 존재하지 않음")
endif()

# include 및 링크 설정
include_directories(${TFLITE_INCLUDE_DIR})

add_executable(yolov5 yolo.cpp main.cpp tf_ocr.cpp plate.cpp)

# 라이브러리 링크
target_link_libraries(yolov5
    PRIVATE
    ${OpenCV_LIBRARIES}  # OpenCV 라이브러리
    ${TFLITE_STATIC_LIB}
    ncnn                # ncnn 라이브러리
    Threads::Threads    # CMake Threads 방식
    rt
    dl
    m
)
