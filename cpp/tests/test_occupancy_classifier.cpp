#include "occupancy_classifier.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
    try {
        // 使用真实模型，检查动态批次和输出概率
        const std::filesystem::path model_path =
            "checkpoints/best_parking_classifier.onnx";

        expect(
            std::filesystem::is_regular_file(model_path),
            "Test model file is missing"
        );

        OccupancyClassifier classifier(model_path);

        // 每个车位有48×48个像素，每个像素有RGB三个值
        const std::size_t values_per_spot = 48 * 48 * 3;

        // 同一个分类器先处理1个，再处理553个，最后切回1个
        const std::vector<std::size_t> batch_sizes{1, 553, 1};

        for (std::size_t count : batch_sizes) {
            std::vector<float> batch(count * values_per_spot, 0.0f);

            const std::vector<float> scores =
                classifier.predict(batch, count);

            expect(
                scores.size() == count,
                "Output count does not match input count"
            );

            for (float score : scores) {
                expect(
                    std::isfinite(score),
                    "Output contains NaN or infinity"
                );

                expect(
                    score >= 0.0f && score <= 1.0f,
                    "Output probability is outside [0, 1]"
                );
            }

            std::cout << "Batch " << count << ": PASSED\n";
        }

        // 拒绝空批次，以及数据长度与车位数量不匹配的输入
        const std::vector<float> one_spot(values_per_spot, 0.0f);

        expect_throws(
            [&] { classifier.predict({}, 0); },
            "Zero spot count should be rejected"
        );

        expect_throws(
            [&] { classifier.predict({}, 1); },
            "Empty input should be rejected"
        );

        expect_throws(
            [&] { classifier.predict(one_spot, 2); },
            "Mismatched input size should be rejected"
        );

        // 确认不存在的模型路径会报错
        const std::filesystem::path missing_model =
            "checkpoints/nonexistent_classifier_for_test.onnx";

        expect(
            !std::filesystem::exists(missing_model),
            "The missing-model test path unexpectedly exists"
        );

        expect_throws(
            [&] { OccupancyClassifier invalid(missing_model); },
            "Missing model should be rejected"
        );

        std::cout << "Occupancy classifier test: PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Occupancy classifier test: FAILED\n"
            << error.what() << '\n';
        return 1;
    }
}