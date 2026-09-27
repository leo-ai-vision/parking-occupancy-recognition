#include "occupancy_classifier.hpp"
#include "parking_config.hpp"
#include "spot_preprocessor.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <nlohmann/json.hpp>

int main(int argc, char* argv[]) {
    // 三个正式输出：视频、状态变化事件、运行统计。
    std::array<std::filesystem::path, 3> final_paths;
    std::array<std::filesystem::path, 3> temporary_paths;
    std::array<bool, 3> owns_temporary{false, false, false};
    std::array<bool, 3> committed{false, false, false};

    try {
        if (argc != 3) {
            std::cerr
                << "Usage: run_video_cpp <input.mp4> <output.mp4>\n";
            return 1;
        }

        const std::filesystem::path input_path = argv[1];
        const std::filesystem::path output_path = argv[2];
        const std::filesystem::path output_dir = output_path.parent_path();
        const std::string output_stem = output_path.stem().string();

        final_paths = {
            output_path,
            output_dir / (output_stem + ".events.jsonl"),
            output_dir / (output_stem + ".stats.json")
        };
        temporary_paths = {
            output_dir / (output_stem + ".partial.mp4"),
            output_dir / (output_stem + ".events.partial.jsonl"),
            output_dir / (output_stem + ".stats.partial.json")
        };

        if (!std::filesystem::is_regular_file(input_path)) {
            throw std::runtime_error("Input video does not exist");
        }
        if (output_path.extension() != ".mp4") {
            throw std::runtime_error("Output must be an .mp4 file");
        }
        if (
            std::filesystem::weakly_canonical(output_path) ==
            std::filesystem::weakly_canonical("docs/demo.mp4")
        ) {
            throw std::runtime_error("Cannot overwrite docs/demo.mp4");
        }

        const ParkingConfig config =
            load_parking_config("configs/parking_spots.json");
        const float threshold =
            load_decision_threshold("configs/decision_threshold.json");
        OccupancyClassifier classifier(
            "checkpoints/best_parking_classifier.onnx"
        );

        cv::VideoCapture video(input_path.string());
        if (!video.isOpened()) {
            throw std::runtime_error("Cannot open input video");
        }

        const double fps = video.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0.0) {
            throw std::runtime_error("Input video FPS is invalid");
        }

        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
        }

        // 任何现有结果都不覆盖；所有输出先写临时文件。
        for (std::size_t i = 0; i < final_paths.size(); ++i) {
            if (std::filesystem::exists(final_paths[i]) ||
                std::filesystem::exists(temporary_paths[i])) {
                throw std::runtime_error(
                    "Output already exists: " + final_paths[i].string()
                );
            }
        }

        owns_temporary[0] = true;
        cv::VideoWriter writer(
            temporary_paths[0].string(),
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            config.image_size
        );
        if (!writer.isOpened()) {
            throw std::runtime_error("Cannot create output video");
        }

        owns_temporary[1] = true;
        std::ofstream events(temporary_paths[1]);
        if (!events) {
            throw std::runtime_error("Cannot create event log");
        }

        std::vector<float> current_scores;
        std::vector<bool> previous_states;
        bool has_previous_states = false;
        std::size_t frame_index = 0;
        std::size_t inference_runs = 0;
        std::size_t event_count = 0;
        double preprocess_ms = 0.0;
        double inference_ms = 0.0;
        const double reported_frames_raw =
            video.get(cv::CAP_PROP_FRAME_COUNT);
        const auto started = std::chrono::steady_clock::now();
        cv::Mat frame;

        while (video.read(frame)) {
            if (frame.empty() || frame.size() != config.image_size) {
                throw std::runtime_error(
                    "Video frame size does not match parking config"
                );
            }

            const cv::Scalar channel_mean = cv::mean(frame);
            const double brightness =
                (channel_mean[0] +
                 channel_mean[1] +
                 channel_mean[2]) / 3.0;

            // 黑场保留在输出视频中，但不进行车位预测
            if (brightness < 70.0) {
                cv::putText(
                    frame,
                    "Waiting for valid video frame",
                    cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(255, 255, 255),
                    2
                );
                writer.write(frame);
                ++frame_index;
                continue;
            }

            // 第一张有效帧立即推理，之后每隔 24 帧更新一次
            if (current_scores.empty() || frame_index % 24 == 0) {
                const auto preprocess_start =
                    std::chrono::steady_clock::now();
                const std::vector<float> batch =
                    prepare_spot_batch(frame, config.spots);
                const auto inference_start =
                    std::chrono::steady_clock::now();
                current_scores =
                    classifier.predict(batch, config.spots.size());
                const auto inference_end =
                    std::chrono::steady_clock::now();

                preprocess_ms += std::chrono::duration<double, std::milli>(
                    inference_start - preprocess_start
                ).count();
                inference_ms += std::chrono::duration<double, std::milli>(
                    inference_end - inference_start
                ).count();
                ++inference_runs;

                // 第一次有效推理只建立基准；后续只记录类别变化。
                std::vector<bool> states(config.spots.size());
                for (std::size_t i = 0; i < states.size(); ++i) {
                    states[i] = current_scores[i] >= threshold;
                    if (has_previous_states &&
                        states[i] != previous_states[i]) {
                        const nlohmann::json event{
                            {"frame_index", frame_index},
                            {"time_seconds",
                             static_cast<double>(frame_index) / fps},
                            {"spot_id", config.spots[i].id},
                            {"old_state",
                             previous_states[i] ? "occupied" : "empty"},
                            {"new_state",
                             states[i] ? "occupied" : "empty"},
                            {"occupied_score", current_scores[i]}
                        };
                        events << event.dump() << '\n';
                        if (!events) {
                            throw std::runtime_error(
                                "Failed to write event log"
                            );
                        }
                        ++event_count;
                    }
                }
                previous_states = std::move(states);
                has_previous_states = true;
            }

            std::size_t occupied_count = 0;
            cv::Mat overlay = frame.clone();
            for (std::size_t i = 0; i < config.spots.size(); ++i) {
                const bool occupied = current_scores[i] >= threshold;
                if (occupied) {
                    ++occupied_count;
                } else {
                    cv::rectangle(
                        overlay,
                        config.spots[i].bbox,
                        cv::Scalar(0, 255, 0),
                        cv::FILLED
                    );
                }
            }

            cv::Mat output_frame;
            cv::addWeighted(overlay, 0.25, frame, 0.75, 0.0, output_frame);
            for (std::size_t i = 0; i < config.spots.size(); ++i) {
                const bool occupied = current_scores[i] >= threshold;

                const cv::Scalar color = occupied
                    ? cv::Scalar(0, 0, 255)
                    : cv::Scalar(0, 255, 0);

                cv::rectangle(
                    output_frame,
                    config.spots[i].bbox,
                    color,
                    2
                );
            }

            cv::putText(
                output_frame,
                "Available: " +
                    std::to_string(config.spots.size() - occupied_count),
                cv::Point(30, 50),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2
            );
            cv::putText(
                output_frame,
                "Occupied: " + std::to_string(occupied_count),
                cv::Point(30, 95),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 0, 255),
                2
            );

            cv::putText(
                output_frame,
                "Total: " + std::to_string(config.spots.size()),
                cv::Point(30, 140),
                cv::FONT_HERSHEY_SIMPLEX,
                0.9,
                cv::Scalar(255, 255, 255),
                2
            );
            cv::putText(
                output_frame,
                "Frame: " + std::to_string(frame_index),
                cv::Point(30, 180),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                2
            );

            writer.write(output_frame);
            ++frame_index;
        }

        writer.release();
        video.release();
        events.close();
        if (!events) {
            throw std::runtime_error("Failed to close event log");
        }

        if (frame_index == 0) {
            throw std::runtime_error("Input video contains no frames");
        }
        if (std::isfinite(reported_frames_raw) &&
            reported_frames_raw > 0.0 &&
            frame_index !=
                static_cast<std::size_t>(std::llround(reported_frames_raw))) {
            throw std::runtime_error(
                "Processed frame count differs from input video metadata"
            );
        }
        if (std::filesystem::file_size(temporary_paths[0]) == 0) {
            throw std::runtime_error("Output video is empty");
        }

        const auto finished = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(
            finished - started
        ).count();
        const double overall_fps = total_ms > 0.0
            ? static_cast<double>(frame_index) * 1000.0 / total_ms
            : 0.0;

        const nlohmann::json stats{
            {"input_video", input_path.string()},
            {"output_video", output_path.string()},
            {"output_events", final_paths[1].string()},
            {"video_fps", fps},
            {"processed_frames", frame_index},
            {"inference_interval", 24},
            {"inference_runs", inference_runs},
            {"event_count", event_count},
            {"parking_spots", config.spots.size()},
            {"decision_threshold", threshold},
            {"preprocess_ms", preprocess_ms},
            {"inference_ms", inference_ms},
            {"average_batch_inference_ms",
             inference_runs > 0
                 ? inference_ms / static_cast<double>(inference_runs)
                 : 0.0},
            {"total_ms", total_ms},
            {"overall_fps", overall_fps}
        };

        owns_temporary[2] = true;
        std::ofstream stats_file(temporary_paths[2]);
        if (!stats_file) {
            throw std::runtime_error("Cannot create statistics file");
        }
        stats_file << stats.dump(2) << '\n';
        stats_file.close();
        if (!stats_file) {
            throw std::runtime_error("Failed to save statistics file");
        }

        // 全部成功后才公布正式结果；若其中一步失败则回滚本次输出。
        for (const auto& path : final_paths) {
            if (std::filesystem::exists(path)) {
                throw std::runtime_error(
                    "Output appeared during processing: " + path.string()
                );
            }
        }
        for (std::size_t i = 0; i < final_paths.size(); ++i) {
            std::filesystem::rename(temporary_paths[i], final_paths[i]);
            owns_temporary[i] = false;
            committed[i] = true;
        }

        std::cout << "Processed frames: " << frame_index << '\n';
        std::cout << "Inference runs: " << inference_runs << '\n';
        std::cout << "State-change events: " << event_count << '\n';
        std::cout << "Overall speed: " << overall_fps << " FPS\n";
        std::cout << "Output video: " << output_path << '\n';
        std::cout << "Events: " << final_paths[1] << '\n';
        std::cout << "Statistics: " << final_paths[2] << '\n';
        return 0;
    } catch (const std::exception& error) {
        for (std::size_t i = 0; i < temporary_paths.size(); ++i) {
            std::error_code ignored;
            if (owns_temporary[i]) {
                std::filesystem::remove(temporary_paths[i], ignored);
            }
            if (committed[i]) {
                std::filesystem::remove(final_paths[i], ignored);
            }
        }
        std::cerr << "Failed: " << error.what() << '\n';
        return 1;
    }
}
