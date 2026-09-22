import json
from pathlib import Path

import matplotlib.pyplot as plt


# 训练历史文件
HISTORY_PATH = Path(
    "results/training/history.json"
)

# 训练曲线保存位置
OUTPUT_PATH = Path(
    "results/training/training_curves.png"
)


def find_best_epoch(validation_auc):
    """
    找出验证集AUC最高的训练轮次。
    """

    best_index = 0

    for index in range(len(validation_auc)):
        if validation_auc[index] > validation_auc[best_index]:
            best_index = index

    return best_index


def main():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"找不到训练历史：{HISTORY_PATH}"
        )

    # 读取每轮训练得到的指标
    with HISTORY_PATH.open(
        "r",
        encoding="utf-8",
    ) as history_file:
        history = json.load(history_file)

    train_auc = history["auc"]
    validation_auc = history["val_auc"]

    train_loss = history["loss"]
    validation_loss = history["val_loss"]

    # 横坐标从第1轮开始，而不是从0开始
    epochs = list(
        range(1, len(train_auc) + 1)
    )

    best_index = find_best_epoch(
        validation_auc
    )

    best_epoch = best_index + 1
    best_auc = validation_auc[best_index]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    # 左图：训练集和验证集AUC
    axes[0].plot(
        epochs,
        train_auc,
        marker="o",
        label="Training AUC",
    )

    axes[0].plot(
        epochs,
        validation_auc,
        marker="o",
        label="Validation AUC",
    )

    # 标出验证集AUC最高的轮次
    axes[0].scatter(
        best_epoch,
        best_auc,
        color="red",
        s=80,
        zorder=5,
        label=f"Best epoch: {best_epoch}",
    )

    axes[0].set_title(
        "Training and Validation AUC"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUC")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # 右图：训练集和验证集损失
    axes[1].plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss",
    )

    axes[1].plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation Loss",
    )

    axes[1].axvline(
        best_epoch,
        color="red",
        linestyle="--",
        label=f"Best epoch: {best_epoch}",
    )

    axes[1].set_title(
        "Training and Validation Loss"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

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

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Training curves saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()