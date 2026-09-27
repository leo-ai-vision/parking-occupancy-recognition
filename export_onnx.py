from pathlib import Path

import onnx
import tensorflow as tf
# 读取已经训练好的 Keras 模型 → 去掉只在训练阶段使用的数据增强层
# → 保留真正的预处理和分类网络 → 导出为 ONNX → 检查 ONNX 文件是否合法。
# ============================================================
# 1. 模型文件路径
# ============================================================
# 已经训练完成并保存好的 Keras 模型
MODEL_PATH = Path("checkpoints/best_parking_classifier.keras")
OUTPUT_PATH = Path("checkpoints/best_parking_classifier.onnx")# ONNX 模型最终保存的位置
# ============================================================
# 2. 主程序
# ============================================================
def main():
    # 2.1 检查 Keras 模型是否存在
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到 Keras 模型：{MODEL_PATH}")
    # 2.2 防止覆盖已经存在的 ONNX 模型
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"ONNX 文件已存在，避免覆盖：{OUTPUT_PATH}")
    # 3. 加载训练好的 Keras 模型
    model = tf.keras.models.load_model(str(MODEL_PATH))
    # ========================================================
    # 4. 创建专门用于推理的模型
    # ========================================================
    # 跳过仅在训练时使用的数据增强层；后面的 Rescaling 和分类权重保持不变
    inference_model = tf.keras.Model(
        inputs=model.get_layer("data_augmentation").output,#data_augmentation 的输出  当成新模型的输入。
        outputs=model.output, # 输出仍然使用原模型最终输出。
    )
    # ========================================================
    # 5. 导出 ONNX 模型
    # ========================================================
    # None 表示批次大小可变，后续才能一次推理 553 个车位
    inference_model.export(
        str(OUTPUT_PATH), # ONNX 文件保存路径
        format="onnx",# 明确要求导出成 ONNX
        # 5.1 定义模型输入格式
        input_signature=[
            tf.TensorSpec(
                shape=(None, 48, 48, 3),
                dtype=tf.float32,
                name="images",
            )
        ],
    )
    # ========================================================
    # 6. 为什么 Batch 维度使用 None？
    # ========================================================
    # 上面的：shape=(None, 48, 48, 3) None 的意思不是“没有 Batch”。而是： Batch 大小可以动态变化。
    #例如：(1, 48, 48, 3)可以一次预测 1 个停车位。也可以：(32, 48, 48, 3)一次预测 32 个。
    #甚至：(553, 48, 48, 3)一次传入停车场全部 553 个车位。所以 None = Dynamic Batch Size。
    # ========================================================
    # 7. 读取刚刚生成的 ONNX 模型
    # ========================================================
    onnx_model = onnx.load(str(OUTPUT_PATH))
    # ========================================================
    # 8. 检查 ONNX 模型是否合法
    # ========================================================
    onnx.checker.check_model(onnx_model)
    print(f"ONNX 模型已导出并通过检查：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()