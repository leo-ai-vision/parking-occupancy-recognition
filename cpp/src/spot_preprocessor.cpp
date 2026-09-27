#include "spot_preprocessor.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>


std::vector<float> prepare_spot_batch(
    const cv::Mat& bgr,
    const std::vector<Spot>& spots
) {
    // 检查输入：必须是非空、二维、8位三通道图像
    if (bgr.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    if (bgr.dims != 2 || bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "Input image must be a 2D CV_8UC3 BGR image"
        );
    }

    if (spots.empty()) {
        throw std::runtime_error("Spot list is empty");
    }

    // 每个车位需要48×48×3个float，提前预留空间
    const std::size_t values_per_spot =
        static_cast<std::size_t>(SPOT_WIDTH) *
        SPOT_HEIGHT * SPOT_CHANNELS;

    std::vector<float> batch;

    if (spots.size() > batch.max_size() / values_per_spot) {
        throw std::runtime_error("Too many spots");
    }

    batch.reserve(spots.size() * values_per_spot);

    // 循环中复用中间图像，减少重复分配内存
    cv::Mat resized;
    cv::Mat rgb;
    cv::Mat float_rgb;

    for (const Spot& spot : spots) {
        const cv::Rect& box = spot.bbox;

        // 检查矩形面积和边界
        // 使用宽度比较，避免直接计算x+width可能造成整数溢出
        if (box.x < 0 || box.y < 0 ||
            box.width <= 0 || box.height <= 0 ||
            box.x > bgr.cols || box.y > bgr.rows ||
            box.width > bgr.cols - box.x ||
            box.height > bgr.rows - box.y) {
            throw std::runtime_error(
                "Invalid crop for spot " + std::to_string(spot.id)
            );
        }

        // 从原图中取得当前车位区域
        const cv::Mat crop = bgr(box);

        // 与Python推理一致，使用双线性插值缩放到48×48
        cv::resize(
            crop,
            resized,
            cv::Size(SPOT_WIDTH, SPOT_HEIGHT),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        // OpenCV使用BGR，模型要求RGB
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // 只转换数据类型，像素值仍然保持0～255
        // 模型内部已经包含除以255的归一化操作
        rgb.convertTo(float_rgb, CV_32FC3);

        // 按行复制到连续数组，顺序为：车位、行、列、RGB
        // 逐行读取也能正确处理带有行间填充的图像内存
        const std::size_t values_per_row =
            static_cast<std::size_t>(SPOT_WIDTH) * SPOT_CHANNELS;

        for (int row = 0; row < SPOT_HEIGHT; ++row) {
            const float* row_data = float_rgb.ptr<float>(row);

            batch.insert(
                batch.end(),
                row_data,
                row_data + values_per_row
            );
        }
    }

    return batch;
}