# `ultralytics/models/yolo/obb/__init__.py`

旋转框（OBB）检测子包入口，再导出三个任务类。

## 逐段说明

### 行 3–5

导入 `OBBPredictor`、`OBBTrainer`、`OBBValidator`；训练/验证继承 `detect` 模块并替换 IoU 与后处理为 probiou/角度维。

### 行 7：`__all__`

导出上述三个名称。