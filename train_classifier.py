import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


# 数据目录
DATASET_ROOT = Path(
    "data/raw/parking_spots"
)

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"

# 模型和训练结果保存路径
CHECKPOINT_PATH = Path(
    "checkpoints/best_parking_classifier.keras"
)

RESULT_DIR = Path(
    "results/training"
)

HISTORY_PATH = RESULT_DIR / "history.json"
METRICS_PATH = RESULT_DIR / "test_metrics.json"
TRAINING_LOG_PATH = RESULT_DIR / "training_log.csv"

# empty必须对应0，occupied对应1
CLASS_NAMES = [
    "empty",
    "occupied",
]

# 统一模型输入尺寸
IMAGE_SIZE = (48, 48)

BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def count_images(class_directory):
    """
    统计类别目录中的有效图片文件数量。
    """

    count = 0

    for file_path in class_directory.iterdir():
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in IMAGE_SUFFIXES
        ):
            count += 1

    return count


def calculate_class_weights():
    """
    根据训练集类别数量计算类别权重。

    empty图片较少，因此训练时给予它更高的损失权重，
    减少模型总是预测occupied的倾向。
    empty:    96
    occupied: 285
    """

    class_counts = []

    for class_name in CLASS_NAMES:
        class_directory = (
            TRAIN_DIR / class_name
        )

        count = count_images(
            class_directory
        )

        if count == 0:
            raise ValueError(
                f"类别目录中没有图片："
                f"{class_directory}"
            )

        class_counts.append(count)

    total_count = sum(class_counts)
    class_number = len(class_counts)

    class_weights = {}

    for class_index, count in enumerate(
        class_counts
    ):
        weight = total_count / (
            class_number * count
        )

        class_weights[
            class_index
        ] = weight

    return class_counts, class_weights


def load_datasets():
    """
    加载训练集、验证集和最终测试集。

    subset="both"表示由同一次操作同时生成训练集和验证集，
    可以保证两个数据集来自同一次随机划分且互不重叠。
    """

    # 一次性完成训练集和验证集划分
    train_dataset, validation_dataset = (
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

    # 独立测试集不参与训练和模型选择
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

    # 自动选择合适的预取数量
    autotune = tf.data.AUTOTUNE

    # 训练当前批次时，提前准备下一批数据
    train_dataset = train_dataset.prefetch(
        buffer_size=autotune
    )

    # 验证集和测试集内容不会随机增强，可以缓存
    validation_dataset = (
        validation_dataset
        .cache()
        .prefetch(buffer_size=autotune)
    )

    test_dataset = (
        test_dataset
        .cache()
        .prefetch(buffer_size=autotune)
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def build_model():
    """
    创建用于空闲/占用识别的轻量级卷积神经网络。
    """

    # 只在训练阶段执行的数据增强
    data_augmentation = models.Sequential(
        [
            # 停车位左右翻转后类别不会改变
            layers.RandomFlip(
                "horizontal"
            ),

            # 模拟坐标框存在少量位置偏差
            layers.RandomTranslation(
                height_factor=0.05,#在水平和垂直方向随机移动最多约5%。
                width_factor=0.05,
            ),

            # 模拟车位距离和裁剪大小变化
            layers.RandomZoom(
                height_factor=0.10,#随机缩放约10%，模拟裁剪区域大小变化。
                width_factor=0.10,
            ),

            # 模拟轻微摄像头倾斜
            layers.RandomRotation(
                factor=5 / 360    #随机旋转
            ),
        ],
        name="data_augmentation",
    )
    # 定义模型输入
    inputs = layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        )
    )
    # 3.应用数据增强
    x = data_augmentation(inputs)
    # 4.像素归一化
    # 将像素值从0～255转换到0～1
    x = layers.Rescaling(
        scale=1.0 / 255
    )(x)

    # 第一层主要学习：
    # - 图像边缘；
    # - 明暗变化；
    # - 简单颜色；
    # - 停车线和车辆局部纹理。
    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)#标准化
    x = layers.ReLU()(x)#负数变成0，正数保持不变，为网络加入非线性能力。
    x = layers.MaxPooling2D()(x)#2×2 池化，把宽高缩小一半：

    # 第二组卷积：提取车辆轮廓和停车位结构
    # - 车辆轮廓；
    # - 车顶纹理；
    # - 挡风玻璃；
    # - 停车位地面结构；
    # - 车辆阴影。
    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # 第三组卷积：低层特征组合成更复杂的特征，用来判断图像中是否存在车辆。
    x = layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # 8.将每个通道压缩成一个特征值：6×6×128 → 128
    x = layers.GlobalAveragePooling2D()(x)
    # 9.全连接层：将128维特征组合成64维高层特征。
    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    # 随机关闭部分神经元，降低过拟合风险：训练时随机将30%的特征设置为0。
    x = layers.Dropout(0.30)(x)

    # 二分类输出：接近0表示empty，接近1表示occupied
    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="parking_occupancy_classifier",
    )
    # 12.构建模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",#二分类交叉熵衡量预测概率与真实标签的差距。
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    return model


def save_history(history):
    """
    将每个训练轮次的指标保存成JSON。
    """

    serializable_history = {}

    for metric_name, values in (
        history.history.items()
    ):
        serializable_history[
            metric_name
        ] = [
            float(value)
            for value in values
        ]

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as history_file:
        json.dump(
            serializable_history,
            history_file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    # 固定随机种子，提高实验可复现性
    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_counts, class_weights = (
        calculate_class_weights()
    )

    print(
        f"Training class counts: "
        f"{dict(zip(CLASS_NAMES, class_counts))}"
    )

    print(
        f"Class weights: {class_weights}"
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = load_datasets()

    model = build_model()
    model.summary()

    callbacks = [
        # 只保存验证集AUC最高的模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        # 验证集长期没有提升时提前结束
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),

        # 验证损失停滞时自动减小学习率
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=0.000001,
            verbose=1,
        ),

        # 将每轮训练指标保存为CSV
        tf.keras.callbacks.CSVLogger(
            filename=TRAINING_LOG_PATH
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    save_history(history)

    # 重新加载验证集AUC最高的模型
    best_model = (
        tf.keras.models.load_model(
            CHECKPOINT_PATH
        )
    )

    # 独立测试集只在训练结束后使用
    test_metrics = best_model.evaluate(
        test_dataset,
        return_dict=True,
    )

    serializable_metrics = {}

    for metric_name, value in (
        test_metrics.items()
    ):
        serializable_metrics[
            metric_name
        ] = float(value)

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            serializable_metrics,
            metrics_file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFinal test metrics")

    for metric_name, value in (
        serializable_metrics.items()
    ):
        print(
            f"  {metric_name}: "
            f"{value:.4f}"
        )

    print(
        f"\nBest model saved to: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Training history saved to: "
        f"{HISTORY_PATH}"
    )

    print(
        f"Test metrics saved to: "
        f"{METRICS_PATH}"
    )


if __name__ == "__main__":
    main()