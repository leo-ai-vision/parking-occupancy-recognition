from pathlib import Path

import cv2
import numpy as np

# 原项目处理第1380帧和第1410帧后，使用第二张图生成最终坐标。
# 新项目显式选择第1410帧作为初始基准，避免依赖文件读取顺序；
# 第1380帧后续用于验证车位坐标在相邻时刻是否稳定。
FRAME_PATH = Path(
    "data/interim/frames/frame_001410.jpg"
)

# 保存坐标检测各阶段的中间结果
OUTPUT_DIR = Path(
    "results/coordinate_detection"
)


def select_bright_regions(image):
    """
    保留图像中的明亮区域。

    停车位标线通常接近白色，因此要求B、G、R
    三个通道的像素值都位于120到255之间。
    """

    # OpenCV读取图像后的通道顺序是BGR
    # 三个通道使用相同阈值，因此这里不需要转换成RGB
    lower_bgr = np.array(
        [120, 120, 120],
        dtype=np.uint8,
    )

    upper_bgr = np.array(
        [255, 255, 255],
        dtype=np.uint8,
    )

    # 满足阈值的像素设为255，不满足的设为0
    bright_mask = cv2.inRange(
        image,
        lower_bgr,
        upper_bgr,
    )

    # 根据二值掩膜保留原图中的明亮区域
    bright_regions = cv2.bitwise_and(
        image,
        image,
        mask=bright_mask,
    )

    return bright_mask, bright_regions


def save_image(output_path, image):
    """
    保存处理结果，并在失败时抛出异常。
    """

    saved = cv2.imwrite(
        str(output_path),
        image,
    )

    if not saved:
        raise RuntimeError(
            f"无法保存图片：{output_path}"
        )

def detect_edges(bright_regions):
    """
    将明亮区域转换成灰度图，再使用Canny检测边缘。

    low_threshold和high_threshold分别是Canny的低阈值和高阈值。
    较强的边缘直接保留，较弱但与强边缘相连的边缘也会保留。
    """

    # 彩色图像转换成单通道灰度图
    gray_image = cv2.cvtColor(
        bright_regions,
        cv2.COLOR_BGR2GRAY,
    )

    # 检测灰度图中的明显边缘
    edge_image = cv2.Canny(
        gray_image,
        50,
        200,
    )

    return gray_image, edge_image


def select_parking_region(edge_image, original_image):
    """
    只保留停车场主体区域，过滤建筑物和道路等无关部分。

    多边形顶点按图像宽高比例计算，因此图像分辨率不变时，
    不需要手工填写每个顶点的像素坐标。
    """

    height, width = edge_image.shape[:2]

    # 按原项目的停车场范围定义六边形ROI
    vertices = np.array(
        [
            [int(width * 0.05), int(height * 0.90)],
            [int(width * 0.05), int(height * 0.70)],
            [int(width * 0.30), int(height * 0.55)],
            [int(width * 0.60), int(height * 0.15)],
            [int(width * 0.90), int(height * 0.15)],
            [int(width * 0.90), int(height * 0.90)],
        ],
        dtype=np.int32,
    )

    # 创建与边缘图大小相同的全黑掩膜
    region_mask = np.zeros_like(edge_image)

    # 将多边形内部填充为白色
    cv2.fillPoly(
        region_mask,
        [vertices],
        255,
    )

    # 只保留多边形范围内的边缘
    region_edges = cv2.bitwise_and(
        edge_image,
        region_mask,
    )

    # 在原图上绘制红色ROI边界，便于检查范围是否合理
    region_preview = original_image.copy()

    cv2.polylines(
        region_preview,
        [vertices],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3,
    )

    return region_edges, region_preview

def detect_horizontal_lines(region_edges):
    """
    使用概率霍夫变换检测线段，再筛选车位分隔线。

    车位分隔线的特点：
    1. 接近水平，即两个端点的y坐标非常接近；
    2. 长度适中，排除过短噪声和过长道路边缘。
    """

    # 检测ROI边缘图中的所有候选线段
    detected_lines = cv2.HoughLinesP(
        region_edges,
        rho=0.1,             # 距离分辨率
        theta=np.pi / 10,    # 角度分辨率为18度
        threshold=15,        # 至少15个边缘点支持才认为是线段
        minLineLength=9,     # 小于9像素的线段直接忽略
        maxLineGap=4,        # 间隔不超过4像素的边缘可连接成一条线     这对停车线很有用，因为车辆遮挡、图像压缩和 Canny 检测可能导致一条停车线出现小断口
    )
#     rho越小 → 距离划分越细
#     rho越大 → 距离划分越粗


    horizontal_lines = []

    # 没有检测到任何线段时，直接返回空列表
    if detected_lines is None:
        return horizontal_lines, 0

    raw_line_count = len(detected_lines)

    for line in detected_lines:
        # HoughLinesP返回的每条线形状为[[x1, y1, x2, y2]]
        x1, y1, x2, y2 = line.reshape(4)

        horizontal_difference = abs(y2 - y1)
        line_length = abs(x2 - x1)

        # y方向差值不超过1像素，认为线段近似水平
        is_horizontal = horizontal_difference <= 1

        # 车位分隔线长度通常位于25到55像素之间
        has_valid_length = 25 <= line_length <= 55

        if is_horizontal and has_valid_length:
            horizontal_lines.append(
                (x1, y1, x2, y2)
            )

    return horizontal_lines, raw_line_count


def draw_detected_lines(image, horizontal_lines):
    """
    在原始图像上绘制筛选后的水平车位线。
    """

    preview = image.copy()

    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(
            preview,
            (x1, y1),
            (x2, y2),
            color=(0, 0, 255),  # BGR格式中的红色
            thickness=2,
        )

    return preview

def cluster_lines_by_x(horizontal_lines, x_tolerance=10):
    """
    按线段起点的x坐标进行连续聚类。

    判断新线段是否属于当前组时，与排序后的上一条线比较。
    这样可以允许车位线的横坐标因透视关系逐渐变化，
    避免同一停车区域被拆成多个小区域。
    """

    if len(horizontal_lines) == 0:
        return []

    # 先按x坐标从左到右排序，
    # x坐标相同时再按y坐标从上到下排序
    sorted_lines = sorted(
        horizontal_lines,
        key=lambda line: (line[0], line[1]),
    )

    clusters = []
    current_cluster = [sorted_lines[0]]

    # 保存排序后上一条线的起点x坐标
    previous_x = sorted_lines[0][0]

    for line in sorted_lines[1:]:
        current_x = line[0]

        # 当前线与上一条线的横向距离
        x_distance = abs(
            current_x - previous_x
        )

        if x_distance <= x_tolerance:
            # 相邻线段的x坐标接近，加入当前聚类
            current_cluster.append(line)
        else:
            # 横向距离过大，结束当前聚类
            clusters.append(current_cluster)

            # 使用当前线开始一个新聚类
            current_cluster = [line]

        # 更新上一条线的x坐标
        previous_x = current_x

    # 添加循环结束后尚未保存的最后一个聚类
    clusters.append(current_cluster)

    return clusters


def build_lane_rectangles(line_clusters, min_line_count=6):
    # 把每组水平线转换成一个纵向停车区域矩形。
    """
    将包含足够多水平线的聚类转换成停车区域矩形。
    少于min_line_count条线的聚类通常是车辆或路面噪声，
    不作为停车区域处理。
    """

    lane_rectangles = []

    for cluster in line_clusters:
        # 删除坐标完全相同的重复线段
        unique_lines = list(set(cluster))
        # 然后过滤线段太少的组：
        if len(unique_lines) < min_line_count:
            continue

        # 按照线段在图像中的纵向位置排序
        unique_lines = sorted(
            unique_lines,
            key=lambda line: line[1],
        )
        # 排序后：第一条线位于区域顶部。最后一条线位于区域底部。


        x1_sum = 0
        x2_sum = 0

        for x1, y1, x2, y2 in unique_lines:
            x1_sum += x1
            x2_sum += x2

        # 再计算所有线段左右端点的平均值：平均后得到的区域边界更稳定。
        average_x1 = int(
            x1_sum / len(unique_lines)
        )

        average_x2 = int(
            x2_sum / len(unique_lines)
        )

        # 第一条和最后一条线的位置作为区域上下边界
        top_y = unique_lines[0][1]
        bottom_y = unique_lines[-1][1]

        lane_rectangles.append(
            (
                average_x1,
                top_y,
                average_x2,
                bottom_y,
            )
        )
    #这仍然是一整列停车区域，不是单个车位。
    # 按照区域的横坐标从左到右排序
    lane_rectangles = sorted(
        lane_rectangles,
        key=lambda rectangle: rectangle[0],
    )

    return lane_rectangles


def draw_lane_rectangles(image, lane_rectangles):
    """
    把检测结果画出来，方便人工检查。
    """

    preview = image.copy()
    # 左右增加7像素缓冲
    horizontal_buffer = 7

    for lane_index, rectangle in enumerate(
        lane_rectangles
    ):
        x1, y1, x2, y2 = rectangle

        top_left = (
            x1 - horizontal_buffer,
            y1,
        )

        bottom_right = (
            x2 + horizontal_buffer,
            y2,
        )

        # 使用绿色矩形显示停车区域
        cv2.rectangle(
            preview,
            top_left,
            bottom_right,
            color=(0, 255, 0),
            thickness=3,
        )

        # 在矩形顶部显示区域编号
        cv2.putText(
            preview,
            str(lane_index),
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return preview



def main():
    # 检查代表帧是否存在
    if not FRAME_PATH.exists():
        raise FileNotFoundError(
            f"找不到代表帧：{FRAME_PATH}"
        )

    # 读取代表帧
    image = cv2.imread(str(FRAME_PATH))

    if image is None:
        raise RuntimeError(
            f"无法读取图片：{FRAME_PATH}"
        )

    # 创建结果输出目录
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 第一步：提取明亮区域
    bright_mask, bright_regions = (
        select_bright_regions(image)
    )

    # 第二步：灰度转换与边缘检测
    gray_image, edge_image = detect_edges(
        bright_regions
    )

    # 第三步：只保留停车场主体区域
    region_edges, region_preview = (
        select_parking_region(
            edge_image,
            image,
        )
    )
    # 第四步：检测并筛选近似水平的车位分隔线
    horizontal_lines, raw_line_count = (
        detect_horizontal_lines(region_edges)
    )

    line_preview = draw_detected_lines(
        image,
        horizontal_lines,
    )
    # 第五步：将横坐标接近的线段聚成停车区域
    line_clusters = cluster_lines_by_x(
        horizontal_lines,
        x_tolerance=10,
    )

    lane_rectangles = build_lane_rectangles(
        line_clusters,
        min_line_count=6,
    )

    lane_preview = draw_lane_rectangles(
        image,
        lane_rectangles,
    )

    # 保存每个处理阶段的结果
    save_image(
        OUTPUT_DIR / "01_bright_mask.png",
        bright_mask,
    )

    save_image(
        OUTPUT_DIR / "02_bright_regions.jpg",
        bright_regions,
    )

    save_image(
        OUTPUT_DIR / "03_gray_image.png",
        gray_image,
    )

    save_image(
        OUTPUT_DIR / "04_edge_image.png",
        edge_image,
    )

    save_image(
        OUTPUT_DIR / "05_region_preview.jpg",
        region_preview,
    )

    save_image(
        OUTPUT_DIR / "06_region_edges.png",
        region_edges,
    )
    save_image(
        OUTPUT_DIR / "07_horizontal_lines.jpg",
        line_preview,
    )
    save_image(
        OUTPUT_DIR / "08_parking_lanes.jpg",
        lane_preview,
    )

    print("Parking-region preprocessing completed.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Raw Hough lines: {raw_line_count}")
    print(
        f"Filtered horizontal lines: "
        f"{len(horizontal_lines)}"
    )
    print(
        f"X-coordinate clusters: "
        f"{len(line_clusters)}"
    )

    print(
        f"Parking lane rectangles: "
        f"{len(lane_rectangles)}"
    )


if __name__ == "__main__":
    main()