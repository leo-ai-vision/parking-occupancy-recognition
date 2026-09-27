#include "parking_config.hpp"
#include "test_util.hpp"

#include <iostream>
#include <cmath>

int main() {
    try {
        // 读取刚才创建的小型测试配置
        const ParkingConfig config = load_parking_config(
            "cpp/tests/fixtures/two_spots.json"
        );

        // 先检查尺寸和数量，确认可以安全访问两个车位
        expect(
            config.image_size == cv::Size(10, 10),
            "Image size should be 10 x 10"
        );

        expect(
            config.spots.size() == 2,
            "There should be exactly 2 spots"
        );

        // 第一个框：[1,2,4,6] -> 位置(1,2)，宽3，高4
        expect(config.spots[0].id == 1, "First spot ID should be 1");
        expect(
            config.spots[0].bbox == cv::Rect(1, 2, 3, 4),
            "First spot rectangle is incorrect"
        );

        // 第二个框：[4,2,8,6] -> 位置(4,2)，宽4，高4
        expect(config.spots[1].id == 2, "Second spot ID should be 2");
        expect(
            config.spots[1].bbox == cv::Rect(4, 2, 4, 4),
            "Second spot rectangle is incorrect"
        );
                // 先确认测试文件存在，避免把“文件缺失”误认为编号检查通过
        const std::filesystem::path duplicate_path =
            "cpp/tests/fixtures/duplicate_id.json";

        expect(
            std::filesystem::is_regular_file(duplicate_path),
            "Duplicate ID fixture file is missing"
        );

        // 将读取操作交给 expect_throws，要求它必须抛出异常
        expect_throws(
            [&duplicate_path]() {
                load_parking_config(duplicate_path);
            },
            "Duplicate spot IDs should be rejected"
        );
                // 检查越界坐标是否被拒绝
        const std::filesystem::path bounds_path =
            "cpp/tests/fixtures/out_of_bounds.json";

        expect(
            std::filesystem::is_regular_file(bounds_path),
            "Out-of-bounds fixture file is missing"
        );

        expect_throws(
            [&bounds_path]() {
                load_parking_config(bounds_path);
            },
            "Out-of-bounds coordinates should be rejected"
        );

        // 读取项目中的真实配置，验证能处理全部553个车位
        const ParkingConfig real_config = load_parking_config(
            "configs/parking_spots.json"
        );

        expect(
            real_config.image_size == cv::Size(1280, 720),
            "Real image size should be 1280 x 720"
        );

        expect(
            real_config.spots.size() == 553,
            "Real config should contain 553 spots"
        );

        std::cout << "Real parking spots: "
                  << real_config.spots.size() << '\n';
                // 使用已知值0.5，检查阈值字段是否被正确读取
        const float test_threshold = load_decision_threshold(
            "cpp/tests/fixtures/threshold_half.json"
        );

        expect(
            test_threshold == 0.5f,
            "Test threshold should be 0.5"
        );

        // 检查真实配置中的阈值，浮点数比较允许很小的误差
        const float real_threshold = load_decision_threshold(
            "configs/decision_threshold.json"
        );

        expect(
            std::abs(real_threshold - 0.017097605392336845) < 1e-8,
            "Real decision threshold is incorrect"
        );

        std::cout << "Decision threshold: "
                  << real_threshold << '\n';

        std::cout << "Parking config test: PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        // 返回非零值，让测试工具知道测试失败
        std::cerr << "Parking config test: FAILED\n";
        std::cerr << error.what() << '\n';
        return 1;
    }
}