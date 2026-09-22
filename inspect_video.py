from pathlib import Path

import cv2


VIDEO_PATH = Path("data/raw/videos/parking_video.mp4")


def main():
    # 检查视频文件是否存在
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"找不到视频：{VIDEO_PATH}")

    # 打开视频
    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    # 读取视频基本信息
    # 读取视频的基本属性
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    fps = video.get(cv2.CAP_PROP_FPS)  # 每秒播放的帧数
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频画面宽度，单位为像素
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频画面高度，单位为像素

    if fps <= 0:
        video.release()
        raise RuntimeError("视频 FPS 无效")

    duration_seconds = frame_count / fps

    video.release()

    # 输出视频信息
    print(f"Video path: {VIDEO_PATH}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.3f}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration_seconds:.2f} seconds")


if __name__ == "__main__":
    main()