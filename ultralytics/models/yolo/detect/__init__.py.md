# `ultralytics/models/yolo/detect/__init__.py`

目标检测子包入口，聚合检测任务的 Predictor、Trainer、Validator，本身不改动 engine 循环逻辑。

## 逐段说明

### 行 3–5：相对导入

从同目录导入 `DetectionPredictor`、`DetectionTrainer`、`DetectionValidator`，三者分别继承 engine 中对应基类并在各自模块中实现检测专用前后处理与 mAP 指标。

### 行 7：`__all__`

导出上述三个符号，供 `yolo.detect.*` 及 `YOLO.task_map["detect"]` 使用。