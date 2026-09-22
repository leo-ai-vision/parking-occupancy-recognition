import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# evaluate_classifier.py生成的指标文件
METRICS_PATH = Path(
    "results/evaluation/metrics.json"
)

# 混淆矩阵图片的保存位置
OUTPUT_PATH = Path(
    "results/evaluation/confusion_matrix.png"
)


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"找不到评估指标：{METRICS_PATH}"
        )

    # 读取最终测试指标
    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as metrics_file:
        metrics = json.load(metrics_file)

    confusion = metrics[
        "confusion_matrix"
    ]

    # 行表示真实类别，列表示预测类别
    confusion_matrix = np.array(
        [
            [
                confusion["true_negative"],
                confusion["false_positive"],
            ],
            [
                confusion["false_negative"],
                confusion["true_positive"],
            ],
        ]
    )

    class_names = [
        "Empty",
        "Occupied",
    ]

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    # 使用蓝色深浅表示样本数量
    image = axis.imshow(
        confusion_matrix,
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_xticks(
        np.arange(len(class_names))
    )

    axis.set_yticks(
        np.arange(len(class_names))
    )

    axis.set_xticklabels(
        class_names
    )

    axis.set_yticklabels(
        class_names
    )

    axis.set_xlabel(
        "Predicted label"
    )

    axis.set_ylabel(
        "True label"
    )

    axis.set_title(
        "Parking Occupancy Confusion Matrix"
    )

    # 根据背景颜色决定文字使用黑色还是白色
    color_threshold = (
        confusion_matrix.max() / 2
    )

    for row_index in range(
        confusion_matrix.shape[0]
    ):
        for column_index in range(
            confusion_matrix.shape[1]
        ):
            value = confusion_matrix[
                row_index,
                column_index,
            ]

            if value > color_threshold:
                text_color = "white"
            else:
                text_color = "black"

            axis.text(
                column_index,
                row_index,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=14,
            )

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Confusion matrix saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()