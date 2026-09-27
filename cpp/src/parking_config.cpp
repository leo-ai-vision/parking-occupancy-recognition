#include "parking_config.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>
#include <cmath>

namespace {

// 读取整数：拒绝小数、字符串，以及超出 int 范围的数值
int read_int(const nlohmann::json& value, const std::string& field) {
    if (!value.is_number_integer()) {
        throw std::runtime_error(field + " must be an integer");
    }

    const double number = value.get<double>();

    if (number < std::numeric_limits<int>::min() ||
        number > std::numeric_limits<int>::max()) {
        throw std::runtime_error(field + " is outside int range");
    }

    return value.get<int>();
}

}  // namespace


ParkingConfig load_parking_config(const std::filesystem::path& path) {
    // 打开配置文件；文件不存在或无权读取时直接报错
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open parking config: " + path.string()
        );
    }

    try {
        // 将文件内容解析为 JSON，at() 会检查字段是否存在
        nlohmann::json document;
        file >> document;

        const auto& image_size = document.at("image_size");

        const int width = read_int(
            image_size.at("width"), "image_size.width"
        );
        const int height = read_int(
            image_size.at("height"), "image_size.height"
        );
        const int spot_count = read_int(
            document.at("spot_count"), "spot_count"
        );

        // 图像尺寸和车位数量都必须为正数
        if (width <= 0 || height <= 0 || spot_count <= 0) {
            throw std::runtime_error(
                "Image dimensions and spot_count must be positive"
            );
        }

        // 确认车位列表的类型，以及声明数量与实际数量是否一致
        const auto& entries = document.at("spots");

        if (!entries.is_array()) {
            throw std::runtime_error("spots must be an array");
        }

        if (entries.size() != static_cast<std::size_t>(spot_count)) {
            throw std::runtime_error("spot_count does not match spots");
        }

        ParkingConfig config;
        config.image_size = cv::Size(width, height);
        config.spots.reserve(entries.size());

        // 集合中每个编号只保留一次，用于发现重复车位编号
        std::unordered_set<int> used_ids;

        for (const auto& entry : entries) {
            const int id = read_int(entry.at("id"), "spot.id");

            if (!used_ids.insert(id).second) {
                throw std::runtime_error(
                    "Duplicate spot ID: " + std::to_string(id)
                );
            }

            // 每个矩形必须提供 x1、y1、x2、y2 四个整数
            const auto& bbox = entry.at("bbox");

            if (!bbox.is_array() || bbox.size() != 4) {
                throw std::runtime_error(
                    "Invalid bbox format for spot " + std::to_string(id)
                );
            }

            const int x1 = read_int(bbox.at(0), "bbox.x1");
            const int y1 = read_int(bbox.at(1), "bbox.y1");
            const int x2 = read_int(bbox.at(2), "bbox.x2");
            const int y2 = read_int(bbox.at(3), "bbox.y2");

            // 矩形必须有正面积，并且完全位于图像范围内
            if (x1 < 0 || y1 < 0 ||
                x2 <= x1 || y2 <= y1 ||
                x2 > width || y2 > height) {
                throw std::runtime_error(
                    "Invalid bbox coordinates for spot " +
                    std::to_string(id)
                );
            }

            // 将两个角点转换为 OpenCV 的位置、宽度和高度
            Spot spot;
            spot.id = id;
            spot.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);

            config.spots.push_back(spot);
        }

        return config;
    } catch (const std::exception& error) {
        // 在错误中附上配置文件路径，方便定位问题
        throw std::runtime_error(
            "Invalid parking config '" + path.string() +
            "': " + error.what()
        );
    }
}

float load_decision_threshold(const std::filesystem::path& path) {
    // 打开阈值配置文件
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open threshold config: " + path.string()
        );
    }

    try {
        nlohmann::json document;
        file >> document;

        // 模型输出表示占用分数，因此正类名称必须是 occupied
        const std::string positive_class =
            document.at("positive_class").get<std::string>();

        if (positive_class != "occupied") {
            throw std::runtime_error(
                "positive_class must be occupied"
            );
        }

        // 阈值必须是数值，拒绝字符串、空值和布尔值
        const auto& value = document.at("threshold");

        if (!value.is_number()) {
            throw std::runtime_error(
                "threshold must be a number"
            );
        }

        const double threshold = value.get<double>();

        // 拒绝无穷大、NaN，以及超出0到1范围的阈值
        if (!std::isfinite(threshold) ||
            threshold < 0.0 || threshold > 1.0) {
            throw std::runtime_error(
                "threshold must be finite and within [0, 1]"
            );
        }

        // 转成与模型输出一致的 float 类型
        return static_cast<float>(threshold);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Invalid threshold config '" + path.string() +
            "': " + error.what()
        );
    }
}