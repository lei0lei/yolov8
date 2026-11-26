# `ultralytics/models/yolo/detect/train.py`

`DetectionTrainer` 是多数 YOLO 检测类任务的训练基类：YOLO 数据集、`DetectionModel`、验证器 `DetectionValidator`，相对 `BaseTrainer` 实现检测 batch 归一化、多尺度、max_det 与数据集对齐。

## 逐段说明

### `__init__`（52–60）

直接 `super().__init__`，无额外 task 覆盖（由调用方或默认 cfg 定 task）。

### `build_dataset`（62–74）

`stride = max(model.stride, 32)`；`build_yolo_dataset(..., rect=mode=="val", stride=gs)`。

### `get_dataloader`（76–103）

`mode` 须 train/val；DDP zero_first 建集；train 可 shuffle，rect+不均匀 batch_shapes 时警告并关 shuffle；workers 在 val 时加倍；`drop_last` 同 compile+train。

### `preprocess_batch`（105–135）

张量字段上 device；`img` 除 255；`multi_scale>0` 时随机 `sz`（stride 对齐），`interpolate` 缩放到新尺寸。

### `set_model_attributes`（137–142）

写 `nc`、`names`、`colors`、`args` 到 model。

### `set_model_names_for_load`（144–148）

`cls_remap` 且 data 有 names 时，load 前设 `model.names` 便于按名 remap 分类头。

### `get_class_counts`（150–155）

训练 labels 按 `source_index` 去重后拼接 `cls` 做 `bincount`（实例级计数）。

### `get_model`（157–173）

`DetectionModel` + 可选 weights。

### `get_validator`（175–179）

`DetectionValidator` 绑定 `test_loader`。

### `_build_train_pipeline`（181–187）

`super()` 后对 detect/segment/pose/obb 调用 `DetectionValidator._check_max_det` 并按 `args.max_det` 设 head 属性。

### `progress_string`（189–197）

检测式进度表头。

### `plot_training_samples`（199–213）

`plot_images`，有 `semantic_mask` 时显示 names（World/YOLOE 提示槽）；`class_colors`。

### `plot_training_labels`（215–219）

拼接全训练集 bbox/cls，`plot_labels`。