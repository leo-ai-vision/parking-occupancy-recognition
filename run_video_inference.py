import json
import time
from pathlib import Path

import cv2
import tensorflow as tf

from predict_frame import (
    draw_predictions,
    load_json,
    prepare_spot_batch,
)


# 输入与输出视频
INPUT_VIDEO_PATH = Path(
    "data/raw/videos/parking_video.mp4"
)

OUTPUT_VIDEO_PATH = Path(
    "results/inference/parking_occupancy_result.mp4"
)

# 保存运行速度等统计信息
STATS_PATH = Path(
    "results/inference/video_inference_stats.json"
)

# 模型、车位坐标和阈值
MODEL_PATH = Path(
    "checkpoints/best_parking_classifier.keras"
)

SPOT_CONFIG_PATH = Path(
    "configs/parking_spots.json"
)

THRESHOLD_PATH = Path(
    "configs/decision_threshold.json"
)

# 约每秒执行一次553个车位的分类
# 视频帧率约为23.976 FPS
INFERENCE_INTERVAL = 24

# None表示处理完整视频；调试时可改成240，仅处理约10秒
MAX_FRAMES = None

# 跳过视频开头的黑场和淡入画面
MIN_MEAN_BRIGHTNESS = 70.0

PREDICTION_BATCH_SIZE = 128
# 默认关闭实时窗口，保证批处理能够完整生成输出视频
# 需要边处理边观看时改为True，并可按Q提前结束
DISPLAY_WINDOW = False

# OpenCV显示窗口名称
WINDOW_NAME = "Parking Occupancy Recognition"

def create_video_writer(
    output_path,
    fps,
    width,
    height,
):
    """
    创建MP4视频写入器。
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    codec = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        codec,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建输出视频："
            f"{output_path}"
        )

    return writer


def main():
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"找不到输入视频："
            f"{INPUT_VIDEO_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}"
        )

    # 加载坐标和分类阈值
    spot_config = load_json(
        SPOT_CONFIG_PATH
    )

    threshold_config = load_json(
        THRESHOLD_PATH
    )

    parking_spots = spot_config["spots"]

    threshold = float(
        threshold_config["threshold"]
    )

    expected_width = spot_config[
        "image_size"
    ]["width"]

    expected_height = spot_config[
        "image_size"
    ]["height"]

    # 加载训练得到的最佳模型
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    video = cv2.VideoCapture(
        str(INPUT_VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"无法打开视频："
            f"{INPUT_VIDEO_PATH}"
        )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_video_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # 固定车位坐标要求视频分辨率保持一致
    if (
        width != expected_width
        or height != expected_height
    ):
        video.release()

        raise ValueError(
            f"视频分辨率为{width}x{height}，"
            f"车位坐标要求"
            f"{expected_width}x{expected_height}"
        )

    writer = create_video_writer(
        OUTPUT_VIDEO_PATH,
        fps,
        width,
        height,
    )

    frame_index = 0
    written_frame_count = 0
    inference_count = 0

    # 保存最近一次推理结果
    # 两次推理之间的帧复用该结果
    current_scores = None

    inference_time_sum = 0.0
    processing_start_time = (
        time.perf_counter()
    )

    try:
        while True:
            success, frame = video.read()

            if not success or frame is None:
                break

            if (
                MAX_FRAMES is not None
                and frame_index >= MAX_FRAMES
            ):
                break

            # 使用平均亮度识别黑场和淡入帧
            mean_brightness = float(
                frame.mean()
            )

            if (
                mean_brightness
                < MIN_MEAN_BRIGHTNESS
            ):
                # 无效帧不执行模型推理
                cv2.putText(
                    frame,
                    "Waiting for valid video frame",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                writer.write(frame)

                frame_index += 1
                written_frame_count += 1
                continue

            should_run_inference = (
                current_scores is None
                or frame_index
                % INFERENCE_INTERVAL
                == 0
            )

            if should_run_inference:
                # 裁剪并准备553个车位
                spot_batch = prepare_spot_batch(
                    frame,
                    parking_spots,
                )

                inference_start_time = (
                    time.perf_counter()
                )

                # 一次批量预测全部车位
                current_scores = model.predict(
                    spot_batch,
                    batch_size=(
                        PREDICTION_BATCH_SIZE
                    ),
                    verbose=0,
                ).reshape(-1)

                inference_elapsed = (
                    time.perf_counter()
                    - inference_start_time
                )

                inference_time_sum += (
                    inference_elapsed
                )

                inference_count += 1

            # 在当前帧上绘制最近一次预测结果
            (
                output_frame,
                available_count,
                occupied_count,
            ) = draw_predictions(
                frame,
                parking_spots,
                current_scores,
                threshold,
            )

            # 显示当前视频帧编号
            cv2.putText(
                output_frame,
                f"Frame: {frame_index}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            writer.write(output_frame)
            # 在OpenCV窗口中实时显示标注结果
            if DISPLAY_WINDOW:
                cv2.imshow(
                    WINDOW_NAME,
                    output_frame,
                )

                # 根据原视频帧率控制播放速度
                display_delay = max(
                    1,
                    round(1000 / fps),
                )

                pressed_key = (
                        cv2.waitKey(display_delay)
                        & 0xFF
                )

                # 按Q键可以提前退出视频处理
                if pressed_key == ord("q"):
                    print(
                        "Video processing stopped "
                        "by user."
                    )
                    break

            frame_index += 1
            written_frame_count += 1

            # 每处理100帧显示一次进度
            if written_frame_count % 100 == 0:
                print(
                    f"Processed "
                    f"{written_frame_count} frames"
                )


    finally:

        # 释放输入视频和输出视频

        video.release()

        writer.release()

        # 关闭OpenCV创建的视频窗口

        if DISPLAY_WINDOW:
            cv2.destroyAllWindows()

    processing_elapsed = (
        time.perf_counter()
        - processing_start_time
    )

    if inference_count > 0:
        average_inference_ms = (
            inference_time_sum
            / inference_count
            * 1000
        )
    else:
        average_inference_ms = 0.0

    if processing_elapsed > 0:
        processing_fps = (
            written_frame_count
            / processing_elapsed
        )
    else:
        processing_fps = 0.0

    stats = {
        "input_video": str(
            INPUT_VIDEO_PATH
        ),
        "output_video": str(
            OUTPUT_VIDEO_PATH
        ),
        "video_fps": float(fps),
        "video_total_frames": (
            total_video_frames
        ),
        "processed_frames": (
            written_frame_count
        ),
        "inference_interval": (
            INFERENCE_INTERVAL
        ),
        "inference_runs": (
            inference_count
        ),
        "average_batch_inference_ms": (
            average_inference_ms
        ),
        "processing_fps": (
            processing_fps
        ),
        "decision_threshold": threshold,
        "parking_spots": len(
            parking_spots
        ),
    }

    STATS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STATS_PATH.open(
        "w",
        encoding="utf-8",
    ) as stats_file:
        json.dump(
            stats,
            stats_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Processed frames: "
        f"{written_frame_count}"
    )
    print(
        f"Inference runs: "
        f"{inference_count}"
    )
    print(
        f"Average batch inference: "
        f"{average_inference_ms:.2f} ms"
    )
    print(
        f"Overall processing speed: "
        f"{processing_fps:.2f} FPS"
    )
    print(
        f"Output video saved to: "
        f"{OUTPUT_VIDEO_PATH}"
    )
    print(
        f"Statistics saved to: "
        f"{STATS_PATH}"
    )


if __name__ == "__main__":
    main()
