import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# 独立测试集
TEST_DIR = Path(
    "data/raw/parking_spots/test"
)

# 已保存的最佳模型
MODEL_PATH = Path(
    "checkpoints/best_parking_classifier.keras"
)

# 验证集选择出的分类阈值
THRESHOLD_PATH = Path(
    "configs/decision_threshold.json"
)

# 误判样本和汇总图保存位置
ERROR_DIR = Path(
    "results/evaluation/error_samples"
)

ERROR_FIGURE_PATH = (
    Path("results/evaluation")
    / "misclassified_samples.png"
)

ERROR_JSON_PATH = (
    Path("results/evaluation")
    / "error_analysis.json"
)

CLASS_NAMES = [
    "empty",
    "occupied",
]

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32


def load_threshold():
    """
    读取验证集选择出的决策阈值。
    """

    with THRESHOLD_PATH.open(
        "r",
        encoding="utf-8",
    ) as threshold_file:
        threshold_config = json.load(
            threshold_file
        )

    return float(
        threshold_config["threshold"]
    )


def collect_predictions(model, dataset):
    """
    按测试集原始顺序收集标签和预测分数。
    """

    labels = []
    scores = []

    for images, batch_labels in dataset:
        batch_scores = model.predict_on_batch(
            images
        )

        labels.extend(
            batch_labels
            .numpy()
            .reshape(-1)
            .astype(int)
            .tolist()
        )

        scores.extend(
            batch_scores
            .reshape(-1)
            .tolist()
        )

    return (
        np.asarray(labels),
        np.asarray(scores),
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"找不到阈值配置：{THRESHOLD_PATH}"
        )

    threshold = load_threshold()

    # shuffle=False保证图片路径、标签和预测结果顺序一致
    test_dataset = (
        tf.keras.utils
        .image_dataset_from_directory(
            TEST_DIR,
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False,
        )
    )

    file_paths = []

    for file_path in test_dataset.file_paths:
        file_paths.append(
            Path(file_path)
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    labels, scores = collect_predictions(
        model,
        test_dataset,
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    error_indices = []

    for index in range(len(labels)):
        if labels[index] != predictions[index]:
            error_indices.append(index)

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 删除上一次运行生成的误判样本副本
    for old_file in ERROR_DIR.glob(
        "error_*"
    ):
        if old_file.is_file():
            old_file.unlink()

    error_records = []

    for error_number, index in enumerate(
        error_indices,
        start=1,
    ):
        source_path = file_paths[index]

        actual_index = int(labels[index])
        predicted_index = int(
            predictions[index]
        )

        actual_name = CLASS_NAMES[
            actual_index
        ]

        predicted_name = CLASS_NAMES[
            predicted_index
        ]

        score = float(scores[index])

        output_name = (
            f"error_{error_number:02d}"
            f"_actual_{actual_name}"
            f"_predicted_{predicted_name}"
            f"_score_{score:.6f}"
            f"{source_path.suffix.lower()}"
        )

        output_path = ERROR_DIR / output_name

        shutil.copy2(
            source_path,
            output_path,
        )

        error_records.append(
            {
                "source_path": str(source_path),
                "saved_path": str(output_path),
                "actual": actual_name,
                "predicted": predicted_name,
                "score": score,
                "threshold": threshold,
            }
        )

    with ERROR_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as error_file:
        json.dump(
            error_records,
            error_file,
            ensure_ascii=False,
            indent=2,
        )

    if len(error_records) > 0:
        column_count = 2
        row_count = (
            len(error_records)
            + column_count - 1
        ) // column_count

        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(
                8,
                row_count * 3.5,
            ),
        )

        axes = np.asarray(
            axes
        ).reshape(-1)

        for plot_index, record in enumerate(
            error_records
        ):
            image = plt.imread(
                record["source_path"]
            )

            axes[plot_index].imshow(image)

            axes[plot_index].set_title(
                f"Actual: {record['actual']}\n"
                f"Predicted: {record['predicted']}\n"
                f"Score: {record['score']:.6f}"
            )

            axes[plot_index].axis("off")

        # 隐藏没有使用的子图
        for plot_index in range(
            len(error_records),
            len(axes),
        ):
            axes[plot_index].axis("off")

        figure.suptitle(
            "Misclassified Test Samples"
        )

        figure.tight_layout()

        figure.savefig(
            ERROR_FIGURE_PATH,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(figure)

    false_positive_count = 0
    false_negative_count = 0

    for record in error_records:
        if (
            record["actual"] == "empty"
            and record["predicted"]
            == "occupied"
        ):
            false_positive_count += 1
        else:
            false_negative_count += 1

    print(f"Decision threshold: {threshold:.6f}")
    print(f"Test samples: {len(labels)}")
    print(f"Misclassified samples: {len(error_records)}")
    print(f"False positives: {false_positive_count}")
    print(f"False negatives: {false_negative_count}")
    print(f"Error samples saved to: {ERROR_DIR}")
    print(
        f"Error figure saved to: "
        f"{ERROR_FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()