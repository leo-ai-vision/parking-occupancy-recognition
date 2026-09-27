#include "parking_config.hpp"
#include "occupancy_classifier.hpp"
#include "spot_preprocessor.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>

int main(int argc, char* argv[]) {
    try {
        // argv[0] 是程序名，后面依次是三个文件路径
        if (argc != 4) {
            std::cerr
                << "Usage: predict_frame_cpp "
                << "<image> <spots.json> <threshold.json>\n";
            return 1;
        }

        const std::filesystem::path image_path = argv[1];
        const std::filesystem::path spots_path = argv[2];
        const std::filesystem::path threshold_path = argv[3];

        // 读取车位坐标、分类阈值和停车场图片
        const ParkingConfig config =
            load_parking_config(spots_path);
        const float threshold =
            load_decision_threshold(threshold_path);
        const cv::Mat image =
            cv::imread(image_path.string(), cv::IMREAD_COLOR);

        if (image.empty()) {
            throw std::runtime_error(
                "Cannot read image: " + image_path.string()
            );
        }

        // 坐标是按固定尺寸标注的，尺寸不同就停止
        if (image.size() != config.image_size) {
            throw std::runtime_error(
                "Image size does not match parking config"
            );
        }

        std::cout << "Image size: "
                  << image.cols << " x " << image.rows << '\n';
        std::cout << "Parking spots: "
                  << config.spots.size() << '\n';
        std::cout << "Decision threshold: "
                  << threshold << '\n';

        // 按配置顺序裁剪车位，生成模型需要的输入
        const std::vector<float> batch =
            prepare_spot_batch(image, config.spots);

        // 加载 ONNX 模型，一次处理整批车位
        OccupancyClassifier classifier(
            "checkpoints/best_parking_classifier.onnx"
        );
        const std::vector<float> scores =
            classifier.predict(batch, config.spots.size());

        if (scores.size() != config.spots.size()) {
            throw std::runtime_error(
                "Number of scores does not match parking spots"
            );
        }

        // 概率达到配置阈值的车位记为占用
        std::size_t occupied_count = 0;
        for (float score : scores) {
            if (score >= threshold) {
                ++occupied_count;
            }
        }

        std::cout << "Inference scores: "
                  << scores.size() << '\n';
        std::cout << "Occupied: "
                  << occupied_count << '\n';
        std::cout << "Available: "
                  << scores.size() - occupied_count << '\n';

        // 将车位 ID 与对应分数组合；map 会按 ID 升序排列
        std::map<int, float> scores_by_id;
        for (std::size_t i = 0; i < scores.size(); ++i) {
            scores_by_id.emplace(config.spots[i].id, scores[i]);
        }

        // 保存逐车位结果，供后续与 Python 逐 ID 对比
        const std::filesystem::path csv_path =
            std::filesystem::path("results/evaluation") /
            ("cpp_predictions_" + image_path.stem().string() + ".csv");

        std::filesystem::create_directories(csv_path.parent_path());
        std::ofstream csv(csv_path);
        if (!csv) {
            throw std::runtime_error(
                "Cannot write CSV: " + csv_path.string()
            );
        }

        csv << "spot_id,occupied_score,occupied\n";
        csv << std::setprecision(9);

        for (const auto& [spot_id, score] : scores_by_id) {
            csv << spot_id << ','
                << score << ','
                << (score >= threshold ? 1 : 0) << '\n';
        }

        csv.close();
        if (!csv) {
            throw std::runtime_error(
                "Failed to save CSV: " + csv_path.string()
            );
        }

        std::cout << "CSV saved to: " << csv_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << '\n';
        return 1;
    }
}