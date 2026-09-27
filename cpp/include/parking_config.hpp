#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>


// 保存一个车位的信息：唯一编号和图像中的矩形位置
struct Spot {
    int id = 0;

    // cv::Rect 使用 x、y、width、height 表示矩形
    // JSON 中的 [x1, y1, x2, y2] 需要在读取时转换
    cv::Rect bbox;
};


// 保存整张停车场图像的尺寸，以及所有车位
struct ParkingConfig {
    // 图像宽度和高度，用于检查车位坐标是否越界
    cv::Size image_size;

    // 动态数组，每个元素保存一个车位
    std::vector<Spot> spots;
};


// 读取车位配置，并检查编号、坐标和车位数量是否合法
ParkingConfig load_parking_config(
    const std::filesystem::path& path
);


// 读取分类阈值，并检查数值范围及正类名称
float load_decision_threshold(
    const std::filesystem::path& path
);