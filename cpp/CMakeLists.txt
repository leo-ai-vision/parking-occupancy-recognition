# 指定最低 CMake 版本，并声明这是一个 C++ 项目
cmake_minimum_required(VERSION 3.20)
project(parking_occupancy_cpp LANGUAGES CXX)

# 使用标准 C++17，关闭编译器特有的语言扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 生成编译命令信息，方便 VS Code 分析代码
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找图像处理、图片读写和视频读写所需的 OpenCV 模块
find_package(
    OpenCV 4 REQUIRED
    COMPONENTS core imgproc imgcodecs videoio
)

# 查找用于读取车位坐标和分类阈值的 JSON 库
find_package(nlohmann_json 3.11 CONFIG REQUIRED)
# 使用 Ubuntu 中已经安装的 ONNX Runtime C++ 开发库
find_package(onnxruntime CONFIG REQUIRED)

# 显示实际找到的配置目录，方便检查依赖位置
message(STATUS "ONNX Runtime config: ${onnxruntime_DIR}")

# 启用测试支持，后续加入配置解析和预处理测试
enable_testing()

# 输出找到的依赖版本，方便确认构建环境
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "JSON version: ${nlohmann_json_VERSION}")
# 将配置读取和图像预处理一起编译到核心库
add_library(
    parking_core STATIC
    src/parking_config.cpp
    src/spot_preprocessor.cpp
    src/occupancy_classifier.cpp
)

# PUBLIC 表示使用这个库的程序也会获得这些头文件路径
target_include_directories(
    parking_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

# 为核心库连接 OpenCV、ONNX Runtime 和 JSON 库
# PUBLIC 依赖也会传递给使用 parking_core 的程序
target_link_libraries(
    parking_core
    PUBLIC
        ${OpenCV_LIBS}
        onnxruntime::onnxruntime
    PRIVATE
        nlohmann_json::nlohmann_json
)

# 编译测试程序，并链接包含函数实现的 parking_core 库
add_executable(test_parking_config tests/test_parking_config.cpp)
target_link_libraries(test_parking_config PRIVATE parking_core)

# 注册测试，运行时从项目根目录查找测试数据
add_test(NAME parking_config COMMAND test_parking_config)

set_tests_properties(
    parking_config PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
# 编译并注册车位图像预处理测试
add_executable(
    test_spot_preprocessor
    tests/test_spot_preprocessor.cpp
)

target_link_libraries(test_spot_preprocessor PRIVATE parking_core)

add_test(NAME spot_preprocessor COMMAND test_spot_preprocessor)
# 编译真实图片预处理数据的导出工具
add_executable(
    dump_preprocessed_batch
    tools/dump_preprocessed_batch.cpp
)

target_link_libraries(dump_preprocessed_batch PRIVATE parking_core)
# 编译并注册分类器测试
add_executable(
    test_occupancy_classifier
    tests/test_occupancy_classifier.cpp
)

target_link_libraries(
    test_occupancy_classifier
    PRIVATE parking_core
)

add_test(
    NAME occupancy_classifier
    COMMAND test_occupancy_classifier
)

# 从项目根目录运行，以便找到 checkpoints 中的模型
set_tests_properties(
    occupancy_classifier PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
add_executable(predict_frame_cpp tools/predict_frame_cpp.cpp)
target_link_libraries(predict_frame_cpp PRIVATE parking_core)
add_executable(run_video_cpp tools/run_video_cpp.cpp)
target_link_libraries(run_video_cpp PRIVATE parking_core nlohmann_json::nlohmann_json)
