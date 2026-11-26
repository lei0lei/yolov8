# `ultralytics/models/yolo/world/__init__.py`

YOLO-World 开放词汇检测训练/验证入口；预测仍复用 `detect.DetectionPredictor`（见 `YOLOWorld.task_map`）。

## 逐段说明

### 行 3–4

- `WorldTrainer`：多模态文本嵌入与 `WorldModel` 训练。
- `WorldValidator`：独立验证时按数据集类名设置 `set_classes`。

### 行 6：`__all__`

仅 `WorldTrainer`、`WorldValidator`（无单独 WorldPredictor）。