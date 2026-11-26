# `ultralytics/models/yolo/semantic/train.py`

`SemanticSegmentationTrainer` 继承 `DetectionTrainer`：`task=semantic`，多边形背景元数据、像素级类权重（ENet 逆对数）、语义验证器。

## 逐段说明

### `__init__`（32–43）

`task=semantic`。

### `get_dataset`（45–47）

`add_polygon_background(super().get_dataset())` 为多边形标注补 background 类元数据。

### `get_model`（49–66）

`SemanticSegmentationModel` + load。

### `get_validator`（68–72）

`SemanticSegmentationValidator`。

### `set_class_weights`（74–77）

`nc>1` 才 `super().set_class_weights()`；二值单通道不加权。

### `get_class_counts`（79–102）

遍历 labels 子集（可选 max_masks 均匀采样）；`load_mask`；`include_class` 过滤；忽略 255；对有效像素 `unique` 计数；返回最后 `nc` 类像素计数。

### `compute_class_weights`（104–107）

`p = count/sum`；权重 `(1/ln(1.02+p))**cls_pw`。

### `plot_training_labels`（109–152）

最多 1000 mask 像素分布柱状图；类名 <30 时 x 轴标注；保存 `labels.jpg`。