# `ultralytics/models/yolo/classify/__init__.py`

分类任务的包入口，仅再导出三个任务类，不继承或修改 engine 基类行为。相对 `BaseTrainer` / `BaseValidator` / `BasePredictor`，具体定制在同级 `predict.py`、`train.py`、`val.py` 中完成。

## 逐段说明

### 行 3–5：类导入

- `ClassificationPredictor`：来自 `predict.py`，扩展 `BasePredictor`。
- `ClassificationTrainer`：来自 `train.py`，扩展 `BaseTrainer`。
- `ClassificationValidator`：来自 `val.py`，扩展 `BaseValidator`。

### 行 7：`__all__`

对外只导出上述三个类名。