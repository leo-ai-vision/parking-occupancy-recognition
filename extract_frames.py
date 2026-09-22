from pathlib import Path

import cv2


# 输入视频路径
VIDEO_PATH = Path("data/raw/videos/parking_video.mp4")

# 提取图片的保存目录
OUTPUT_DIR = Path("data/interim/frames")

# 从视频不同时间位置抽取代表帧，
# 用于验证固定车位坐标在整段视频中是否稳定
FRAME_NUMBERS = [
    60,     # 约2.5秒，避开开头黑场和淡入过程
    360,    # 约15秒
    720,    # 约30秒
    1080,   # 约45秒
    1380,   # 原项目测试帧
    1410,   # 坐标标定基准帧
]


def extract_frame(video, frame_number, output_path):
    """
    从视频中提取指定帧并保存。

    参数：
        video: 已经打开的 OpenCV 视频对象
        frame_number: 从 1 开始的帧编号
        output_path: 图片保存路径
    """

    # OpenCV 的帧索引从 0 开始，因此要减 1
    frame_index = frame_number - 1

    # 将视频读取位置移动到目标帧
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # 读取当前位置的图像
    success, frame = video.read()

    # 如果读取失败，立即报告错误
    if not success or frame is None:
        raise RuntimeError(
            f"无法读取第 {frame_number} 帧，OpenCV 索引为 {frame_index}"
        )

    # 将图像保存到指定位置
    saved = cv2.imwrite(str(output_path), frame)

    # 检查图片是否成功写入磁盘
    if not saved:
        raise RuntimeError(f"无法保存图片：{output_path}")

    print(
        f"Saved frame {frame_number} "
        f"(OpenCV index {frame_index}): {output_path}"
    )


def main():
    # 检查输入视频是否存在
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"找不到视频：{VIDEO_PATH}")

    # 如果输出目录不存在，则自动创建
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    video = cv2.VideoCapture(str(VIDEO_PATH))

    # 检查视频是否成功打开
    if not video.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    # 获取视频总帧数
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        # 依次提取配置中的目标帧
        for frame_number in FRAME_NUMBERS:
            # 合法的帧编号范围是 1 到视频总帧数
            if frame_number < 1 or frame_number > frame_count:
                raise ValueError(
                    f"帧编号 {frame_number} 超出范围，"
                    f"有效范围是 1 到 {frame_count}"
                )

            # 文件名继续使用人类习惯的帧编号
            output_path = OUTPUT_DIR / f"frame_{frame_number:06d}.jpg"

            # 提取并保存当前帧
            extract_frame(video, frame_number, output_path)
    finally:
        # 无论程序是否发生异常，都释放视频资源
        video.release()


if __name__ == "__main__":
    main()
