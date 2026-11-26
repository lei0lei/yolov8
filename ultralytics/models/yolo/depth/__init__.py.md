# `ultralytics/models/yolo/depth/__init__.py`

深度估计子包入口。`calibrate.py` 含标定工具函数，但不纳入 `__all__`；训练阶段由 `DepthTrainer.final_eval` 内部调用标定逻辑。

## 逐段说明

### 行 3–5：导出类

- `DepthPredictor`：深度图推理。
- `DepthTrainer`：继承 `DetectionTrainer`，改损失与验证器。
- `DepthValidator`：无 NMS，使用 `DepthMetrics`。

### 行 7：`__all__`

仅包含上述三个任务类。