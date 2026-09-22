from collections import Counter
from pathlib import Path
import hashlib

import cv2


# 已经标注好的车位分类数据集
DATASET_ROOT = Path(
    "data/raw/parking_spots"
)

# 数据集应当包含的划分和类别
SPLITS = ["train", "test"]
CLASS_NAMES = ["empty", "occupied"]

# 只检查这些常见图片格式
IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def calculate_file_hash(file_path):
    """
    计算文件的SHA-256摘要。

    如果训练集与测试集中存在完全相同的图片，
    它们会得到相同的摘要，可以用于检查数据泄漏。
    """

    file_bytes = file_path.read_bytes()

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def main():
    # 保存各数据划分和类别的图片数量
    class_counts = {}

    # 记录图片尺寸及其出现次数
    image_sizes = Counter()

    # 记录无法读取的损坏图片
    invalid_images = []

    # 分别保存训练集和测试集的文件摘要
    split_hashes = {
        "train": set(),
        "test": set(),
    }

    total_count = 0

    for split in SPLITS:
        for class_name in CLASS_NAMES:
            class_dir = (
                DATASET_ROOT
                / split
                / class_name
            )

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"找不到数据目录：{class_dir}"
                )

            image_paths = []

            for file_path in class_dir.iterdir():
                if (
                    file_path.is_file()
                    and file_path.suffix.lower()
                    in IMAGE_SUFFIXES
                ):
                    image_paths.append(
                        file_path
                    )

            class_counts[
                (split, class_name)
            ] = len(image_paths)

            total_count += len(image_paths)

            for image_path in image_paths:
                image = cv2.imread(
                    str(image_path)
                )

                if image is None:
                    invalid_images.append(
                        str(image_path)
                    )
                    continue

                height, width = image.shape[:2]

                image_sizes[
                    (width, height)
                ] += 1

                file_hash = calculate_file_hash(
                    image_path
                )

                split_hashes[split].add(
                    file_hash
                )

    # 找出同时出现在训练集和测试集中的文件摘要
    duplicate_hashes = (
        split_hashes["train"]
        & split_hashes["test"]
    )

    print("Dataset distribution")

    for split in SPLITS:
        split_total = 0

        print(f"\n{split.upper()}")

        for class_name in CLASS_NAMES:
            count = class_counts[
                (split, class_name)
            ]

            split_total += count

            print(
                f"  {class_name}: {count}"
            )

        print(f"  total: {split_total}")

    print(f"\nAll images: {total_count}")
    print(
        f"Invalid images: "
        f"{len(invalid_images)}"
    )
    print(
        f"Cross-split duplicate files: "
        f"{len(duplicate_hashes)}"
    )

    print("\nImage sizes")

    for image_size, count in sorted(
        image_sizes.items()
    ):
        width, height = image_size

        print(
            f"  {width} x {height}: "
            f"{count}"
        )


if __name__ == "__main__":
    main()