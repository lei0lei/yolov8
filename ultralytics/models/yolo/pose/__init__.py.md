# `ultralytics/models/yolo/pose/__init__.py`

姿态估计子包入口。

## 逐段说明

### 行 3–5

导出 `PosePredictor`、`PoseTrainer`、`PoseValidator`；在检测流水线基础上增加关键点维度的构造结果与 OKS 指标。

### 行 7：`__all__`

三个姿态任务类。