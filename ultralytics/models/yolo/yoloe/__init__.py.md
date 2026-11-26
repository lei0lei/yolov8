# `ultralytics/models/yolo/yoloe/__init__.py`

YOLOE 开放/可提示检测与分割的扩展训练器、验证器与视觉提示预测器集合入口。

## 逐段说明

### 行 3–6：导入

- `predict`：`YOLOEVPDetectPredictor`、`YOLOEVPSegPredictor`。
- `train`：多种 YOLOE 训练变体（标准、PE 线性探测、从零、视觉提示等）。
- `train_seg`：分割侧训练器 Mixin。
- `val`：`YOLOEDetectValidator`、`YOLOESegValidator`（文本/视觉提示验证）。

### 行 8–22：`__all__`

列出上述全部 13 个公开符号，供高级训练脚本与 `YOLOE.task_map` 引用。