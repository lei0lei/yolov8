# `examples/python/Axelera/yolo11-seg.py`

本示例在不依赖 Ultralytics 运行时的情况下，用 Axelera Voyager SDK 的 `axelera.runtime` 管线 API 跑 YOLO11 实例分割。`build_pipeline` 串联颜色空间、letterbox、张量化、解码分割头、并行 NMS 与 mask 重建，最终交给 `ax_segmentation` 与内置 display 应用绘制掩膜与框。支持摄像头索引、视频文件与静图；可选无头模式定期打印检测统计，或通过 OpenCV 渲染器保存带标注视频。模块级 `IMAGE_SUFFIXES` 用于区分单帧图像输入以在推理后保持窗口开启。

## 逐段说明

### `build_pipeline`（第 31–56 行）

参数 `model_path`、`conf`、`iou`。`op.seq` 内：`BGR→RGB`、`letterbox(640)`、`to_tensor`、`load`、Yolo11 `decode_segmentation`（80 类、32 mask 系数）。`par` 分支对检测做 `nms`，另一分支取 proto；再 `par` 将检测映射回原图空间并 `proto_to_mask`，输出接 `ax_segmentation(CocoClasses)`。末尾 `.optimized()` 融合算子。

### `main`（第 59–91 行）

根据 `--no-display` 选 renderer；有 `--output` 时 `expose_surface=True` 供 `save_output_video`。数字 `source` 映射为 `/dev/videoN` 且 backend 用 opencv，否则 ffmpeg。循环 `pipeline.stream(source)`：写帧到窗口或每 100 帧无头打印类别计数。静图且非 headless 时在结束后 `wait_for_close`。

### `__main__` 参数解析（第 94–107 行）

定义 model/source/conf/iou/no-display/output；若同时 `--output` 与 `--no-display` 则 `parser.error` 阻止无效组合。