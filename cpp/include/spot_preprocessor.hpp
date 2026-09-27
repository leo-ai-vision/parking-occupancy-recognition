#pragma once

#include "parking_config.hpp"

#include <vector>
#include <opencv2/core.hpp>


// 模型要求每张车位图片为48×48，包含RGB三个通道
inline constexpr int SPOT_WIDTH = 48;
inline constexpr int SPOT_HEIGHT = 48;
inline constexpr int SPOT_CHANNELS = 3;


// 按spots中的顺序裁剪车位，并生成连续的模型输入数组
//
// 输入：OpenCV读取的8位、三通道BGR图像，以及车位坐标
// 输出：float数组，依次排列为车位、行、列、RGB通道
//
// 像素值保持在0～255，模型内部已经包含归一化操作
// 空图、错误图像类型、空车位列表或越界坐标应抛出异常
std::vector<float> prepare_spot_batch(
    const cv::Mat& bgr,
    const std::vector<Spot>& spots
);