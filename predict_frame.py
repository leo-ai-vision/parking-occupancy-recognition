import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# 用于测试完整推理流程的图片
FRAME_PATH = Path(
    "data/interim/frames/frame_001410.jpg"
)

# 车位坐标、模型和分类阈值
SPOT_CONFIG_PATH = Path(
    "configs/parking_spots.json"
)

THRESHOLD_PATH = Path(
    "configs/decision_threshold.json"
)

MODEL_PATH = Path(
    "checkpoints/best_parking_classifier.keras"
)

# 推理结果保存位置
OUTPUT_PATH = Path(
    "results/inference/frame_001410_prediction.jpg"
)

MODEL_IMAGE_SIZE = (48, 48)
PREDICTION_BATCH_SIZE = 128


def load_json(file_path):
    """
    读取JSON文件，并在文件不存在时给出明确错误。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def prepare_spot_batch(image, parking_spots):
    """
    根据553个车位坐标裁剪图片，
    并生成一次批量预测所需的输入数组。
    """

    spot_images = []

    for spot in parking_spots:
        x1, y1, x2, y2 = spot["bbox"]

        # 从完整停车场图像中裁剪当前车位
        spot_image = image[
            y1:y2,
            x1:x2,
        ]

        if spot_image.size == 0:
            raise ValueError(
                f"车位 {spot['id']} "
                f"裁剪结果为空：{spot['bbox']}"
            )

        # 将不同大小的车位统一缩放到模型输入尺寸
        spot_image = cv2.resize(
            spot_image,
            MODEL_IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

        # OpenCV读取的是BGR，而Keras训练数据使用RGB
        spot_image = cv2.cvtColor(
            spot_image,
            cv2.COLOR_BGR2RGB,
        )

        spot_images.append(
            spot_image
        )

    # 模型内部已经包含Rescaling层，
    # 因此这里只转换数据类型，不再除以255
    batch = np.asarray(
        spot_images,
        dtype=np.float32,
    )

    return batch


def draw_predictions(
    image,
    parking_spots,
    scores,
    threshold,
):
    """
    根据模型预测结果绘制车位状态。
    """

    result = image.copy()
    overlay = image.copy()

    available_count = 0
    occupied_count = 0

    # 第一次绘制：给空闲车位添加绿色填充
    for spot, score in zip(
        parking_spots,
        scores,
    ):
        x1, y1, x2, y2 = spot["bbox"]

        is_occupied = score >= threshold

        if is_occupied:
            occupied_count += 1
        else:
            available_count += 1

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color=(0, 255, 0),
                thickness=-1,
            )

    # 将绿色区域与原图进行半透明融合
    cv2.addWeighted(
        overlay,
        0.45,
        result,
        0.55,
        0,
        result,
    )

    # 第二次绘制：显示所有车位边界
    for spot, score in zip(
        parking_spots,
        scores,
    ):
        x1, y1, x2, y2 = spot["bbox"]

        is_occupied = score >= threshold

        if is_occupied:
            # 红色边框表示占用
            color = (0, 0, 255)
        else:
            # 绿色边框表示空闲
            color = (0, 255, 0)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=1,
        )

    # 使用黑色背景提高文字可读性
    cv2.rectangle(
        result,
        (20, 15),
        (340, 125),
        color=(0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        result,
        f"Available: {available_count}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        result,
        f"Occupied: {occupied_count}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        result,
        f"Total: {len(parking_spots)}",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return (
        result,
        available_count,
        occupied_count,
    )


def main():
    if not FRAME_PATH.exists():
        raise FileNotFoundError(
            f"找不到测试帧：{FRAME_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到分类模型：{MODEL_PATH}"
        )

    # 加载车位坐标和分类阈值
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

    # 读取完整停车场图片
    image = cv2.imread(
        str(FRAME_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"无法读取测试帧：{FRAME_PATH}"
        )

    image_height, image_width = (
        image.shape[:2]
    )

    expected_size = spot_config[
        "image_size"
    ]

    # 固定坐标要求推理图片与标定图片分辨率相同
    if (
        image_width != expected_size["width"]
        or image_height != expected_size["height"]
    ):
        raise ValueError(
            f"图片分辨率为 "
            f"{image_width}x{image_height}，"
            f"坐标要求 "
            f"{expected_size['width']}x"
            f"{expected_size['height']}"
        )

    # 加载训练得到的最佳模型
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # 将553个车位一次性组成批次
    spot_batch = prepare_spot_batch(
        image,
        parking_spots,
    )

    # 批量预测全部车位
    scores = model.predict(
        spot_batch,
        batch_size=PREDICTION_BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    # 根据验证集选择出的阈值生成最终分类
    (
        result,
        available_count,
        occupied_count,
    ) = draw_predictions(
        image,
        parking_spots,
        scores,
        threshold,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = cv2.imwrite(
        str(OUTPUT_PATH),
        result,
    )

    if not saved:
        raise RuntimeError(
            f"无法保存推理结果：{OUTPUT_PATH}"
        )

    print(
        f"Decision threshold: "
        f"{threshold:.6f}"
    )
    print(
        f"Available parking spots: "
        f"{available_count}"
    )
    print(
        f"Occupied parking spots: "
        f"{occupied_count}"
    )
    print(
        f"Total parking spots: "
        f"{len(parking_spots)}"
    )
    print(
        f"Prediction saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()