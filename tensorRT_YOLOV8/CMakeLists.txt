# 设置最低的 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(TensorRTDemo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 设置 TensorRT 和 CUDA 路径
set(TENSORRT_INCLUDE_DIR "/usr/local/tensorrt/include")
set(TENSORRT_LIBRARY_DIR "/usr/lib/aarch64-linux-gnu")
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set(CUDA_LIBRARY_DIR "/usr/local/cuda/lib64")

# 添加包含路径
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置源文件
set(SOURCES
    main.cpp
    InferenceEngine.cpp
    camera.cpp
    tensorRTModel.cpp
)

# 创建可执行文件
add_executable(trt_demo ${SOURCES})

# 链接 TensorRT 和 CUDA 库
target_link_libraries(trt_demo
    ${TENSORRT_LIBRARY_DIR}/libnvinfer.so
    ${TENSORRT_LIBRARY_DIR}/libnvonnxparser.so
    ${CUDA_LIBRARY_DIR}/libcudart.so
    ${OpenCV_LIBS}
)
