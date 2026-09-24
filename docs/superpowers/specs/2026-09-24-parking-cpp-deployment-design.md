# 停车场占用识别的 Linux/C++ 推理部署设计

日期：2026-09-24

## 目标与边界

把已完成的固定摄像头停车场项目扩展为可在 Linux 上构建、运行和测量的 C++17 视频推理程序。Python 继续负责训练、模型导出与独立测试集评估。C++ 负责读取视频、加载 553 个车位坐标、批量裁剪与分类、绘制结果、输出完整视频及带时间戳的车位状态事件。

这个版本用于展示 C++ 软件工程、图像处理、模型部署、自动化事件提取和性能分析。它不使用真实车辆遥测、地理坐标或道路网络，因此文档与简历不得称其为真实地图/GIS、车队数据或自动驾驶导航系统。基于停车场通道图的路径规划可以作为下一阶段独立功能，在核心推理验收后再设计。

## 已有输入和不可改动的事实

- 原视频：1280 × 720、约 23.976 FPS、1452 帧。
- `configs/parking_spots.json` 存储 553 个车位，`bbox` 为图像像素坐标 `[x1, y1, x2, y2]`。
- `checkpoints/best_parking_classifier.keras` 是 TensorFlow/Keras 二分类模型；输出是 `occupied` 概率。
- `configs/decision_threshold.json` 的固定阈值约为 0.017098，仅来自验证集选择。
- 现有 Python 推理对每个车位先按像素框裁剪，采用 OpenCV `INTER_LINEAR` 缩放到 48 × 48，BGR 转 RGB，转 `float32`，保留 0–255 像素范围；模型内部含 `Rescaling(1/255)`。
- 原视频推理每 24 帧更新一次分类结果，对低于平均亮度 70 的淡入帧不推理，并在更新间隔复用最近一次结果。
- 已报告的 97.56% 测试准确率与 68.97 FPS 均来自 Python/TensorFlow 版本。C++ 版本必须重新测量，不能沿用这些性能值。

用户当前在原仓库中有未提交的推理文件和统计结果。实现时保留这些改动；只提交本功能明确修改的文件。

## 实现路径

采用 **Keras 导出 ONNX + ONNX Runtime C++**。不重训现有 CNN，也不把网络逐层重写成 C++。首先使用 Keras 3 的 `Model.export(..., format="onnx")` 导出推理图，再用 Python ONNX Runtime 验证模型输入、输出与预测一致性。只有通过验证的 ONNX 文件才能进入 C++ 视频程序。

备选方案是用 `tf2onnx` 转换 SavedModel；仅在 Keras 导出失败且能定位兼容问题时使用。OpenCV DNN 与 LibTorch 暂不作为主路线，因为前者可能改变算子兼容性，后者要求转换或重训现有 TensorFlow 模型。

## 组件与接口

| 组件 | 职责 | 输入 / 输出 |
| --- | --- | --- |
| `export_onnx.py` | 从 `.keras` 导出 ONNX，检查输入输出形状 | Keras 模型 → ONNX 文件 |
| `verify_onnx.py` | 对测试图像与代表帧比较 Keras/ONNX 概率和类别 | 两种模型、图像 → JSON 一致性报告 |
| `cpp/parking_config` | 读取车位 JSON 和阈值 JSON；检查 schema、ID 与边界 | JSON → 车位列表、分辨率、阈值 |
| `cpp/spot_preprocessor` | 按与 Python 相同的规则生成连续 NHWC `float32` 批量输入 | BGR 帧、车位列表 → `[N,48,48,3]` |
| `cpp/occupancy_classifier` | 负责 ONNX Runtime 会话及批量推理 | ONNX 模型、批量输入 → N 个概率 |
| `cpp/video_pipeline` | 视频读写、每 24 帧推理、状态复用、计时和错误处理 | 视频与配置 → 标注视频、统计 JSON |
| `cpp/event_writer` | 在相邻有效推理结果的类别变化时写状态事件 | 帧号、时间戳、车位、概率 → JSONL |

C++ 程序通过命令行显式传入 `--video`、`--model`、`--spots`、`--threshold`、`--output-video`、`--output-events` 和 `--output-stats`。相对路径以运行时工作目录解析。程序在启动时验证所有输入文件、视频分辨率、视频 FPS、553 个车位的唯一 ID、正面积且不越界的坐标，以及位于 `[0,1]` 的阈值。错误写入标准错误流并返回非零退出码。

输出 ONNX 模型应接受动态批次的 NHWC `float32` 输入 `[N,48,48,3]`，输出 `[N,1]` 的 `occupied` 概率。C++ 端不得再次除以 255。单个批次包含全部 553 个车位；若导出模型只支持固定批次，必须先解决导出问题，不通过逐车位推理掩盖它。

JSONL 事件每行包含 `frame_index`、`time_seconds`、`spot_id`、`old_state`、`new_state` 和 `occupied_score`。第一次有效推理建立初始状态，不产生伪造的“状态变化”；以后仅当相邻推理的类别变化时写事件。时间戳用 `frame_index / video_fps` 计算。事件源是视频分类结果，不称为车辆硬件遥测。

默认无 GUI，在 Linux 服务器上直接写视频。输出路径若已存在则报错；正常完成后才把临时视频文件重命名为正式输出，避免中途退出覆盖完整结果。计时 JSON 分别记录已处理帧数、推理次数、预处理耗时、模型推理耗时、总耗时及整体 FPS；所有时间用 `std::chrono::steady_clock` 统计。

## 构建和依赖

使用 C++17、CMake、OpenCV 4、ONNX Runtime C++ SDK 和 nlohmann/json。`cpp/CMakeLists.txt` 接收显式的 ONNX Runtime SDK 路径，不在构建时偷偷下载依赖。README 提供 Ubuntu/WSL 的依赖安装、Release 构建、运行和结果检查命令，并说明需要兼容的 ONNX Runtime 版本。新增的 C++ 模块配中文块级注释和明确的错误信息。

## 验收方式

1. 导出的 ONNX 文件经 `onnx.checker` 验证，可用 Python ONNX Runtime 对批次大小 1、164 和 553 推理；输入输出形状符合上述契约。
2. 对独立测试集的 164 张图像，ONNX 与 Keras 预测类别逐张一致，最大概率绝对误差不超过 `1e-4`，ONNX 的混淆矩阵与当前报告一致；若阈值附近样本不一致，先定位原因再调整实现，不重新用测试集挑阈值。
3. 对代表帧 `frame_001410.jpg`，Python 与 C++ 预处理输出的每个通道数值最大绝对差不超过 1 个 0–255 像素级，最终 553 个类别与空闲/占用计数一致。
4. Linux Release 构建成功并通过配置解析、坐标边界、预处理、状态事件的单元测试。
5. C++ 程序处理完整 1452 帧视频，输出文件可解码、帧数与原视频一致；有效帧的空闲数与占用数之和恒为 553。
6. 单独报告 C++ 环境的模型批量推理延迟和整体处理 FPS，并注明 CPU、操作系统、编译模式和采样方法；以达到原视频 23.976 FPS 为实时目标，达不到时先剖析瓶颈。

## 暂不包含

不更换 ResNet、不重训模型、不实现真实车队数据接入、不增加云端服务或图形界面、不声称跨摄像头泛化。停车场通道图和最短路径引导属于后续独立阶段；只有形成可验证的图、入口与车位映射，并完成路径测试后才可写入简历。
