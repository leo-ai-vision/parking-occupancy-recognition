#include "occupancy_classifier.hpp"
#include "spot_preprocessor.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

OccupancyClassifier::OccupancyClassifier(
    const std::filesystem::path& model_path
)
    : env_(ORT_LOGGING_LEVEL_WARNING, "parking_occupancy") {
    // 加载模型前先检查文件，避免路径错误难以定位
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error(
            "Model file does not exist: " + model_path.string()
        );
    }

    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    // 本项目的模型必须只有一个输入和一个输出
    if (session_.GetInputCount() != 1 ||
        session_.GetOutputCount() != 1) {
        throw std::runtime_error(
            "Model must have exactly one input and one output"
        );
    }

    // 检查输入：动态批次、48×48、RGB、float32
    const Ort::TypeInfo input_type = session_.GetInputTypeInfo(0);
    if (input_type.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error("Model input is not a tensor");
    }

    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto input_shape = input_info.GetShape();

    if (input_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_shape.size() != 4 ||
        input_shape[0] != -1 ||
        input_shape[1] != SPOT_HEIGHT ||
        input_shape[2] != SPOT_WIDTH ||
        input_shape[3] != SPOT_CHANNELS) {
        throw std::runtime_error(
            "Model input must be dynamic [N,48,48,3] float32"
        );
    }

    // 检查输出：每个车位对应一个float32占用概率
    const Ort::TypeInfo output_type = session_.GetOutputTypeInfo(0);
    if (output_type.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error("Model output is not a tensor");
    }

    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    const auto output_shape = output_info.GetShape();

    if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape.size() != 2 ||
        output_shape[0] != -1 ||
        output_shape[1] != 1) {
        throw std::runtime_error(
            "Model output must be dynamic [N,1] float32"
        );
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // 复制名称到std::string，避免临时名称释放后留下无效指针
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
}

std::vector<float> OccupancyClassifier::predict(
    const std::vector<float>& batch,
    std::size_t spot_count
) {
    const std::size_t values_per_spot =
        SPOT_WIDTH * SPOT_HEIGHT * SPOT_CHANNELS;

    // 检查车位数与输入数据量，避免无效形状或整数溢出
    if (spot_count == 0 ||
        spot_count >
            static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()
            ) / values_per_spot ||
        batch.size() != spot_count * values_per_spot) {
        throw std::runtime_error(
            "Input size does not match spot count"
        );
    }

    const std::array<std::int64_t, 4> shape{
        static_cast<std::int64_t>(spot_count),
        SPOT_HEIGHT,
        SPOT_WIDTH,
        SPOT_CHANNELS
    };

    // ONNX Runtime使用可写数据指针；复制后再创建输入张量
    std::vector<float> input_data = batch;
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        shape.data(),
        shape.size()
    );

    const std::array<const char*, 1> input_names{
        input_name_.c_str()
    };
    const std::array<const char*, 1> output_names{
        output_name_.c_str()
    };

    // 整批车位只调用一次模型推理
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        input_names.size(),
        output_names.data(),
        output_names.size()
    );

    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("Model returned an invalid output");
    }

    const auto result_info =
        outputs[0].GetTensorTypeAndShapeInfo();
    const auto result_shape = result_info.GetShape();

    if (result_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        result_shape.size() != 2 ||
        result_shape[0] !=
            static_cast<std::int64_t>(spot_count) ||
        result_shape[1] != 1) {
        throw std::runtime_error(
            "Model returned an unexpected output shape or type"
        );
    }

    const float* data = outputs[0].GetTensorData<float>();
    std::vector<float> scores(data, data + spot_count);

    for (float score : scores) {
        if (!std::isfinite(score) ||
            score < 0.0f ||
            score > 1.0f) {
            throw std::runtime_error(
                "Model returned an invalid probability"
            );
        }
    }

    return scores;
}