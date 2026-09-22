import json
from pathlib import Path

import cv2

from detect_parking_lines import (
    build_lane_rectangles,
    cluster_lines_by_x,
    detect_edges,
    detect_horizontal_lines,
    select_bright_regions,
    select_parking_region,
)


# 用于生成车位坐标的基准帧
REFERENCE_FRAME_PATH = Path(
    "data/interim/frames/frame_001410.jpg"
)

# 将最终车位坐标保存成JSON，方便检查和后续读取
SPOT_CONFIG_PATH = Path(
    "configs/parking_spots.json"
)

# 保存所有车位的可视化结果
PREVIEW_PATH = Path(
    "results/coordinate_detection/09_parking_spots.jpg"
)

# 相邻车位在图像纵向上的间隔
# 这是基于当前固定摄像头视角得到的标定参数
SPOT_HEIGHT = 15.5


# 每个停车区域的边界微调参数
#
# 自动检测得到的是整列停车区域的大致范围。
# 由于透视变化、车辆遮挡和标线不完整，
# 每一列的上下左右边界还需要进行少量校正。
LANE_ADJUSTMENTS = [
    {"top": 20,  "bottom": 30,  "left": -8,  "right": 0},
    {"top": -10, "bottom": 50,  "left": -15, "right": 15},
    {"top": 0,   "bottom": 15,  "left": -15, "right": 15},
    {"top": -11, "bottom": 10,  "left": -15, "right": 15},
    {"top": 28,  "bottom": -15, "left": -15, "right": 15},
    {"top": 5,   "bottom": 15,  "left": -15, "right": 15},
    {"top": -15, "bottom": 15,  "left": -15, "right": 15},
    {"top": -15, "bottom": -20, "left": -15, "right": 15},
    {"top": -10, "bottom": 15,  "left": -10, "right": 10},
    {"top": -30, "bottom": 15,  "left": -10, "right": 10},
    {"top": 9,   "bottom": 0,   "left": -10, "right": 10},
    {"top": -32, "bottom": 30,  "left": 0,   "right": 0},
]


def detect_lane_rectangles(image):
    """
    调用前面已经实现的图像处理流程，
    从基准帧中检测12个纵向停车区域。
    """

    # 提取停车位的白色标线
    _, bright_regions = select_bright_regions(
        image
    )

    # 转换为灰度图并检测边缘
    _, edge_image = detect_edges(
        bright_regions
    )

    # 只保留停车场主体ROI
    region_edges, _ = select_parking_region(
        edge_image,
        image,
    )

    # 检测并筛选近似水平的车位分隔线
    horizontal_lines, _ = detect_horizontal_lines(
        region_edges
    )

    # 将横坐标接近的线段聚成停车区域
    line_clusters = cluster_lines_by_x(
        horizontal_lines,
        x_tolerance=10,
    )

    # 根据每组线段建立停车区域矩形
    lane_rectangles = build_lane_rectangles(
        line_clusters,
        min_line_count=6,
    )

    return lane_rectangles


def adjust_lane_rectangle(rectangle, lane_index):
    """
    根据对应区域的标定参数微调矩形边界。
    """

    x1, y1, x2, y2 = rectangle

    # 将NumPy整数转换成普通Python整数
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    adjustment = LANE_ADJUSTMENTS[
        lane_index
    ]

    # 对停车区域的四条边分别进行微调
    adjusted_x1 = x1 + adjustment["left"]
    adjusted_y1 = y1 + adjustment["top"]
    adjusted_x2 = x2 + adjustment["right"]
    adjusted_y2 = y2 + adjustment["bottom"]

    # 防止产生宽度或高度小于等于0的无效区域
    if (
        adjusted_x1 >= adjusted_x2
        or adjusted_y1 >= adjusted_y2
    ):
        raise ValueError(
            f"第 {lane_index + 1} 个停车区域"
            f"调整后的坐标无效"
        )

    return (
        adjusted_x1,
        adjusted_y1,
        adjusted_x2,
        adjusted_y2,
    )


def split_lane_into_spots(
    lane_rectangle,
    lane_index,
    lane_count,
):
    """
    将一个纵向停车区域切分成多个独立车位。

    第一列和最后一列是单排车位；
    中间的停车区域左右各有一排车位，
    因此需要再沿区域中线分成左右两列。
    """

    x1, y1, x2, y2 = lane_rectangle

    # 根据停车区域高度计算纵向车位数量
    row_count = int(
        abs(y2 - y1) // SPOT_HEIGHT
    ) + 1

    # 第一列和最后一列为单排，其余区域为双排
    is_single_column = lane_index in (
        0,
        lane_count - 1,
    )

    spots = []

    for row_index in range(row_count):
        # 计算当前车位的上边界
        spot_y1 = int(
            y1 + row_index * SPOT_HEIGHT
        )

        # 计算当前车位的下边界
        spot_y2 = int(
            spot_y1 + SPOT_HEIGHT
        )

        if is_single_column:
            # 单排区域直接使用整个区域宽度
            spots.append(
                {
                    "row": row_index + 1,
                    "side": "single",
                    "bbox": [
                        x1,
                        spot_y1,
                        x2,
                        spot_y2,
                    ],
                }
            )
        else:
            # 双排区域以中间位置分成左右两列
            middle_x = int(
                (x1 + x2) / 2
            )

            # 左侧车位
            spots.append(
                {
                    "row": row_index + 1,
                    "side": "left",
                    "bbox": [
                        x1,
                        spot_y1,
                        middle_x,
                        spot_y2,
                    ],
                }
            )

            # 右侧车位
            spots.append(
                {
                    "row": row_index + 1,
                    "side": "right",
                    "bbox": [
                        middle_x,
                        spot_y1,
                        x2,
                        spot_y2,
                    ],
                }
            )

    return spots


def build_parking_spots(
    lane_rectangles,
    image_width,
    image_height,
):
    """
    对所有停车区域进行微调和切分，
    生成带有唯一编号的车位列表。
    """

    expected_lane_count = len(
        LANE_ADJUSTMENTS
    )

    # 微调参数数量必须与检测到的停车区域数量一致
    if len(lane_rectangles) != expected_lane_count:
        raise RuntimeError(
            f"期望检测到 {expected_lane_count} 个停车区域，"
            f"实际检测到 {len(lane_rectangles)} 个"
        )

    parking_spots = []
    lane_count = len(lane_rectangles)

    for lane_index, rectangle in enumerate(
        lane_rectangles
    ):
        # 根据当前区域的标定参数修正边界
        adjusted_rectangle = (
            adjust_lane_rectangle(
                rectangle,
                lane_index,
            )
        )

        # 将当前区域切分成单独车位
        lane_spots = split_lane_into_spots(
            adjusted_rectangle,
            lane_index,
            lane_count,
        )

        for spot in lane_spots:
            x1, y1, x2, y2 = spot["bbox"]

            # 检查车位坐标是否位于图像范围内
            if not (
                0 <= x1 < x2 <= image_width
                and 0 <= y1 < y2 <= image_height
            ):
                raise ValueError(
                    f"发现超出图像范围的车位："
                    f"{spot['bbox']}"
                )

            # 为每个车位添加稳定的唯一编号和区域编号
            spot["id"] = len(parking_spots) + 1
            spot["lane"] = lane_index + 1

            parking_spots.append(spot)

    return parking_spots


def draw_parking_spots(image, parking_spots):
    """
    在基准图像上画出全部车位，便于人工检查。
    """

    preview = image.copy()

    for spot in parking_spots:
        x1, y1, x2, y2 = spot["bbox"]

        # 使用蓝色细线显示每个独立车位
        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            color=(255, 0, 0),
            thickness=1,
        )

    # 在图像左上角显示车位总数
    cv2.putText(
        preview,
        f"Total parking spots: {len(parking_spots)}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    return preview


def save_spot_config(
    parking_spots,
    image_width,
    image_height,
):
    """
    将车位坐标及相关元数据保存成JSON文件。
    """

    config = {
        "reference_frame": str(
            REFERENCE_FRAME_PATH
        ),
        "image_size": {
            "width": image_width,
            "height": image_height,
        },
        "spot_height": SPOT_HEIGHT,
        "spot_count": len(parking_spots),
        "spots": parking_spots,
    }

    # 如果configs目录不存在，则自动创建
    SPOT_CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SPOT_CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            config,
            config_file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    # 检查基准帧是否存在
    if not REFERENCE_FRAME_PATH.exists():
        raise FileNotFoundError(
            f"找不到基准帧："
            f"{REFERENCE_FRAME_PATH}"
        )

    # 读取基准帧
    image = cv2.imread(
        str(REFERENCE_FRAME_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"无法读取基准帧："
            f"{REFERENCE_FRAME_PATH}"
        )

    image_height, image_width = image.shape[:2]

    # 从图像中检测12个停车区域
    lane_rectangles = detect_lane_rectangles(
        image
    )

    # 将停车区域切分成独立车位
    parking_spots = build_parking_spots(
        lane_rectangles,
        image_width,
        image_height,
    )

    # 保存JSON坐标配置
    save_spot_config(
        parking_spots,
        image_width,
        image_height,
    )

    # 生成并保存车位可视化结果
    preview = draw_parking_spots(
        image,
        parking_spots,
    )

    PREVIEW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = cv2.imwrite(
        str(PREVIEW_PATH),
        preview,
    )

    if not saved:
        raise RuntimeError(
            f"无法保存车位预览图："
            f"{PREVIEW_PATH}"
        )

    print(
        f"Parking lane rectangles: "
        f"{len(lane_rectangles)}"
    )
    print(
        f"Total parking spots: "
        f"{len(parking_spots)}"
    )
    print(
        f"Saved spot config: "
        f"{SPOT_CONFIG_PATH}"
    )
    print(
        f"Saved spot preview: "
        f"{PREVIEW_PATH}"
    )


if __name__ == "__main__":
    main()