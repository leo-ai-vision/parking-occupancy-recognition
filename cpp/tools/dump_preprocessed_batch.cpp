#include "parking_config.hpp"
#include "spot_preprocessor.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>


int main(int argc, char* argv[]) {
    try {
        // 程序需要三个参数：图片路径、车位配置、输出文件
        // argc还包含程序自身名称，因此总数应为4
        if (argc != 4) {
            std::cerr
                << "Usage: dump_preprocessed_batch "
                << "<image> <spots.json> <output.f32>\n";
            return 1;
        }

        const std::filesystem::path image_path = argv[1];
        const std::filesystem::path config_path = argv[2];
        const std::filesystem::path output_path = argv[3];

        // 读取真实车位配置和停车场图片
        const ParkingConfig config = load_parking_config(config_path);

        const cv::Mat image = cv::imread(
            image_path.string(),
            cv::IMREAD_COLOR
        );

        if (image.empty()) {
            throw std::runtime_error(
                "Cannot read image: " + image_path.string()
            );
        }

        // 坐标对应特定分辨率，尺寸不一致时不能直接使用
        if (image.size() != config.image_size) {
            throw std::runtime_error(
                "Image size does not match parking config"
            );
        }

        const std::vector<float> batch = prepare_spot_batch(
            image, config.spots
        );

        // 保存float32原始数据，后续让Python读取并比较
        static_assert(sizeof(float) == 4, "float must be 4 bytes");

        if (output_path.extension() != ".f32") {
            throw std::runtime_error(
                "Output filename must end with .f32"
            );
        }

        if (!output_path.parent_path().empty()) {
            std::filesystem::create_directories(
                output_path.parent_path()
            );
        }

        std::ofstream output(output_path, std::ios::binary);

        if (!output.is_open()) {
            throw std::runtime_error(
                "Cannot open output: " + output_path.string()
            );
        }

        // 将连续的float数组按字节写入文件
        const std::size_t byte_count = batch.size() * sizeof(float);

        output.write(
            reinterpret_cast<const char*>(batch.data()),
            static_cast<std::streamsize>(byte_count)
        );
        output.close();

        if (!output) {
            throw std::runtime_error(
                "Failed to save batch: " + output_path.string()
            );
        }

        std::cout << "Parking spots: " << config.spots.size() << '\n';
        std::cout << "Float values: " << batch.size() << '\n';
        std::cout << "Output bytes: " << byte_count << '\n';
        std::cout << "Saved batch: " << output_path.string() << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}