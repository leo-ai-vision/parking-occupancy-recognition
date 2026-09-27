#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

// 停车位分类器：管理模型，并对一批车位图像执行推理
class OccupancyClassifier {
public:
    // 创建分类器时加载模型，并检查模型的输入输出格式
    explicit OccupancyClassifier(
        const std::filesystem::path& model_path
    );

    // batch：预处理生成的连续 RGB 浮点数，保持 0～255
    // spot_count：车位数量，输入形状为 [spot_count, 48, 48, 3]
    // 返回值：每个车位的占用概率，顺序与输入车位一致
    std::vector<float> predict(
        const std::vector<float>& batch,
        std::size_t spot_count
    );

private:
    // 推理环境必须比模型会话更早创建、更晚释放
    // 成员按声明顺序初始化，按相反顺序销毁
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};

    // 保存从模型中读取的输入和输出名称
    std::string input_name_;
    std::string output_name_;
};