import json
from pathlib import Path

import cv2


# 保存抽样帧的目录
FRAME_DIR = Path(
    "data/interim/frames"
)

# generate_parking_spots.py生成的车位坐标文件
SPOT_CONFIG_PATH = Path(
    "configs/parking_spots.json"
)

# 坐标验证图片的输出目录
OUTPUT_DIR = Path(
    "results/coordinate_validation"
)


def load_spot_config():
    """
    从JSON文件读取车位坐标和图像尺寸信息。
    """

    if not SPOT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"找不到车位坐标文件："
            f"{SPOT_CONFIG_PATH}"
        )

    with SPOT_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    # 检查JSON中是否包含后续需要的字段
    required_keys = [
        "image_size",
        "spot_count",
        "spots",
    ]

    for key in required_keys:
        if key not in config:
            raise KeyError(
                f"车位配置缺少字段：{key}"
            )

    # 检查记录的数量是否与实际列表长度一致
    if config["spot_count"] != len(
        config["spots"]
    ):
        raise ValueError(
            "JSON中的spot_count与实际车位数量不一致"
        )

    return config


def draw_parking_spots(
    image,
    parking_spots,
    frame_name,
):
    """
    把相同的车位坐标画到一张视频抽样帧上。
    """

    preview = image.copy()

    for spot in parking_spots:
        # bbox按照左、上、右、下的顺序保存
        x1, y1, x2, y2 = spot["bbox"]

        # 使用蓝色细线显示车位范围
        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            color=(255, 0, 0),
            thickness=1,
        )

    # 显示当前验证帧的文件名
    cv2.putText(
        preview,
        f"Frame: {frame_name}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # 显示车位总数
    cv2.putText(
        preview,
        f"Parking spots: {len(parking_spots)}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    return preview


def main():
    # 加载已经生成的车位坐标
    config = load_spot_config()

    parking_spots = config["spots"]

    expected_width = config[
        "image_size"
    ]["width"]

    expected_height = config[
        "image_size"
    ]["height"]

    # 查找frames目录中的所有抽样帧
    frame_paths = sorted(
        FRAME_DIR.glob("frame_*.jpg")
    )

    if len(frame_paths) == 0:
        raise FileNotFoundError(
            f"在目录中没有找到抽样帧："
            f"{FRAME_DIR}"
        )

    # 创建验证结果目录
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for frame_path in frame_paths:
        # 读取当前抽样帧
        image = cv2.imread(
            str(frame_path)
        )

        if image is None:
            raise RuntimeError(
                f"无法读取图片：{frame_path}"
            )

        image_height, image_width = (
            image.shape[:2]
        )

        # 固定坐标只能直接用于相同分辨率的视频帧
        if (
            image_width != expected_width
            or image_height != expected_height
        ):
            raise ValueError(
                f"{frame_path.name}的分辨率为"
                f"{image_width}x{image_height}，"
                f"期望分辨率为"
                f"{expected_width}x{expected_height}"
            )

        # 将553个车位画到当前帧上
        preview = draw_parking_spots(
            image,
            parking_spots,
            frame_path.name,
        )

        # 为验证结果增加_spots后缀
        output_path = (
            OUTPUT_DIR
            / f"{frame_path.stem}_spots.jpg"
        )

        saved = cv2.imwrite(
            str(output_path),
            preview,
        )

        if not saved:
            raise RuntimeError(
                f"无法保存验证图片："
                f"{output_path}"
            )

        saved_count += 1

        print(
            f"Validated {frame_path.name}: "
            f"{output_path}"
        )

    print(
        f"Coordinate validation completed: "
        f"{saved_count} frames"
    )
    print(
        f"Parking spots per frame: "
        f"{len(parking_spots)}"
    )
    print(
        f"Results saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()