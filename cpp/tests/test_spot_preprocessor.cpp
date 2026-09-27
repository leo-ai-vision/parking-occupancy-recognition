#include "spot_preprocessor.hpp"
#include "test_util.hpp"

#include <iostream>


int main() {
    try {
        // 构造2行4列的小图：左半边和右半边使用不同颜色
        // OpenCV中的颜色顺序是B、G、R
        cv::Mat image(2, 4, CV_8UC3);

        image(cv::Rect(0, 0, 2, 2)).setTo(
            cv::Scalar(10, 20, 30)
        );
        image(cv::Rect(2, 0, 2, 2)).setTo(
            cv::Scalar(40, 50, 240)
        );

        // 将左右两块区域分别作为两个车位
        const std::vector<Spot> spots = {
            {1, cv::Rect(0, 0, 2, 2)},
            {2, cv::Rect(2, 0, 2, 2)}
        };

        const std::vector<float> batch = prepare_spot_batch(
            image, spots
        );

        // 每个车位应输出48×48×3个数值
        const std::size_t values_per_spot = 48 * 48 * 3;

        expect(
            batch.size() == 2 * values_per_spot,
            "Batch size is incorrect"
        );

        // 检查全部像素：先放第一个车位，再放第二个车位
        // 每个像素必须变成RGB顺序，并保持0～255的数值
        for (std::size_t i = 0; i < values_per_spot; i += 3) {
            expect(
                batch[i] == 30.0f &&
                batch[i + 1] == 20.0f &&
                batch[i + 2] == 10.0f,
                "First spot RGB values are incorrect"
            );

            const std::size_t offset = values_per_spot + i;

            expect(
                batch[offset] == 240.0f &&
                batch[offset + 1] == 50.0f &&
                batch[offset + 2] == 40.0f,
                "Second spot RGB values are incorrect"
            );
        }

        // 空图、灰度图和空车位列表都应被拒绝
        expect_throws(
            [&]() { prepare_spot_batch(cv::Mat(), spots); },
            "Empty image should be rejected"
        );

        const cv::Mat gray(2, 4, CV_8UC1, cv::Scalar(100));

        expect_throws(
            [&]() { prepare_spot_batch(gray, spots); },
            "Grayscale image should be rejected"
        );

        expect_throws(
            [&]() { prepare_spot_batch(image, {}); },
            "Empty spot list should be rejected"
        );

        // 图像宽度为4，这个框的右边界为5，属于越界
        const std::vector<Spot> invalid_spots = {
            {3, cv::Rect(3, 0, 2, 2)}
        };

        expect_throws(
            [&]() { prepare_spot_batch(image, invalid_spots); },
            "Out-of-bounds crop should be rejected"
        );

        std::cout << "Spot preprocessing test: PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Spot preprocessing test: FAILED\n";
        std::cerr << error.what() << '\n';
        return 1;
    }
}