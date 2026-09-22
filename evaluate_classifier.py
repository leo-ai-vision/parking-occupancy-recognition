import json
from pathlib import Path

import numpy as np
import tensorflow as tf
# 模型能区分empty和occupied，
# 但默认0.5阈值不合适。找到合适的阈值
# ============================================================
# 1. 路径配置
# ============================================================
# 数据路径
TRAIN_DIR = Path(
    "data/raw/parking_spots/train"
)

TEST_DIR = Path(
    "data/raw/parking_spots/test"
)

# 最佳模型路径
MODEL_PATH = Path(
    "checkpoints/best_parking_classifier.keras"
)

# 保存验证集选择出的分类阈值
THRESHOLD_PATH = Path(
    "configs/decision_threshold.json"
)

# 保存最终测试结果
RESULT_DIR = Path(
    "results/evaluation"
)

METRICS_PATH = RESULT_DIR / "metrics.json"
# ============================================================
# 2. 类别配置
# ============================================================
CLASS_NAMES = [
    "empty",
    "occupied",
]

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

# ============================================================
# 3. 加载验证集和测试集
# ============================================================

def load_validation_and_test_datasets():
    """
    重新创建与训练阶段完全相同的验证集，
    同时加载保持独立的测试集。
    """

    _, validation_dataset = (
        tf.keras.utils
        .image_dataset_from_directory(
            TRAIN_DIR,
            class_names=CLASS_NAMES,
            validation_split=VALIDATION_SPLIT,
            subset="both",
            seed=RANDOM_SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=True,
        )
    )
    # --------------------------------------------------------
    # 加载测试集
    # --------------------------------------------------------

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

    return validation_dataset, test_dataset

# ============================================================
# 4. 收集真实标签和模型预测概率
# ============================================================
def collect_predictions(model, dataset):
    """
    收集一个数据集的真实标签和模型预测概率。

    在同一个循环中同时保存标签和预测结果，
    避免随机数据顺序导致二者错位。
    """

    labels = []
    scores = []
    # 真实标签： [0, 1, 1, 0]
    # 预测概率： [0.014, 0.028, 0.023, 0.016]
    # 虽然所有概率都小于0.5，但occupied的概率总体高于empty，因此模型依然具备分类能力。

    for images, batch_labels in dataset:
        # 使用模型预测当前 batch
        batch_scores = model.predict_on_batch(
            images
        )
        # 保存真实标签
        labels.extend(
            batch_labels
            .numpy()# Tensor -> NumPy
            .reshape(-1)# 拉平成一维
            .astype(int)# 0.0/1.0 -> 0/1
            .tolist() # NumPy -> Python list
        )
        # 保存预测概率
        scores.extend(
            batch_scores
            .reshape(-1)
            .tolist()
        )

    return (
        np.asarray(labels),
        np.asarray(scores),
    )
# ============================================================
# 5. 根据指定阈值计算分类指标
# ============================================================
def calculate_metrics(
    labels,
    scores,
    threshold,
):
    # 预测概率 -> 最终类别
    predictions = (
        scores >= threshold
    ).astype(int)
    # ========================================================
    # 混淆矩阵
    #
    #                   预测
    #              empty   occupied
    #
    # 实际 empty      TN       FP
    # 实际 occupied   FN       TP
    #
    # ========================================================
    true_negative = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )

    false_positive = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )

    false_negative = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )

    true_positive = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )

    accuracy = (
        true_positive + true_negative
    ) / len(labels)

    if true_positive + false_positive > 0:
        precision = true_positive / (
            true_positive + false_positive
        )
    else:
        precision = 0.0

    if true_positive + false_negative > 0:
        recall = true_positive / (
            true_positive + false_negative
        )
    else:
        recall = 0.0

    if true_negative + false_positive > 0:
        specificity = true_negative / (
            true_negative + false_positive
        )
    else:
        specificity = 0.0

    balanced_accuracy = (
        recall + specificity
    ) / 2

    if precision + recall > 0:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1_score = 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }
# ============================================================
# 6. 在验证集上寻找最佳阈值
# ============================================================
def select_threshold(labels, scores):
    """
    只使用验证集选择分类阈值。

    选择平衡准确率最高的阈值，
    使empty和occupied两个类别受到同等重视。
    """
    # 构造候选阈值
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                scores,
                np.array([1.0]),
            ]
        )
    )

    best_metrics = None
    # 遍历所有候选阈值
    for threshold in candidate_thresholds:
        metrics = calculate_metrics(
            labels,
            scores,
            threshold,
        )

        if best_metrics is None:
            best_metrics = metrics
            continue

        if (
            metrics["balanced_accuracy"]
            > best_metrics[
                "balanced_accuracy"
            ]
        ):
            best_metrics = metrics

    return best_metrics
# ============================================================
# 7. 计算 AUC
# ============================================================
def calculate_auc(labels, scores):
    """
    计算不依赖固定阈值的AUC。
    """

    auc_metric = tf.keras.metrics.AUC()
    # 输入真实标签和预测分数
    auc_metric.update_state(
        labels,
        scores,
    )

    return float(
        auc_metric.result().numpy()
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到最佳模型：{MODEL_PATH}"
        )

    validation_dataset, test_dataset = (
        load_validation_and_test_datasets()
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # 收集验证集结果并选择阈值
    validation_labels, validation_scores = (
        collect_predictions(
            model,
            validation_dataset,
        )
    )

    validation_metrics = select_threshold(
        validation_labels,
        validation_scores,
    )

    selected_threshold = (
        validation_metrics["threshold"]
    )

    # 使用验证集选出的阈值评估独立测试集
    test_labels, test_scores = (
        collect_predictions(
            model,
            test_dataset,
        )
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_scores,
        selected_threshold,
    )

    test_metrics["auc"] = calculate_auc(
        test_labels,
        test_scores,
    )

    THRESHOLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_config = {
        "positive_class": "occupied",
        "selection_dataset": "validation",
        "selection_metric": (
            "balanced_accuracy"
        ),
        "threshold": selected_threshold,
        "validation_metrics": (
            validation_metrics
        ),
    }

    with THRESHOLD_PATH.open(
        "w",
        encoding="utf-8",
    ) as threshold_file:
        json.dump(
            threshold_config,
            threshold_file,
            ensure_ascii=False,
            indent=2,
        )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            test_metrics,
            metrics_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.6f}"
    )

    print("\nValidation metrics")

    for name, value in (
        validation_metrics.items()
    ):
        if name != "confusion_matrix":
            print(f"  {name}: {value:.4f}")

    print("\nFinal test metrics")

    for name, value in test_metrics.items():
        if name != "confusion_matrix":
            print(f"  {name}: {value:.4f}")

    print(
        "  confusion_matrix: "
        f"{test_metrics['confusion_matrix']}"
    )

    print(
        f"\nThreshold saved to: "
        f"{THRESHOLD_PATH}"
    )

    print(
        f"Metrics saved to: "
        f"{METRICS_PATH}"
    )


if __name__ == "__main__":
    main()