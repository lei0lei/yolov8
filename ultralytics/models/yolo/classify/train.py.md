# `ultralytics/models/yolo/classify/train.py`

`ClassificationTrainer` 扩展 `BaseTrainer`：数据为 `ClassificationDataset`、输出层按 `nc` reshape、验证用 `ClassifyMetrics`，不做检测式 mosaic/多尺度（`multi_scale=0`）。

## 逐段说明

### `__init__`（51–65）

`overrides` 默认 `{}`；强制 `task=classify`、`multi_scale=0.0`；非 resume 且未指定 `imgsz` 时默认 224；再 `super().__init__`。

### `set_model_attributes`（67–69）

`self.model.names = self.data["names"]`。

### `get_class_counts`（71–74）

从 `train_loader.dataset.samples` 按源图索引去重，对类标 `bincount` 得每类图像数（用于类别权重等）。

### `get_model`（76–98）

- 建 `ClassificationModel(nc, ch, verbose)`。
- 可选 `load(weights)`。
- 未 `pretrained` 且非 resume 时对含 `reset_parameters` 的模块重置。
- `Dropout` 且 `args.dropout` 时设置 `m.p`。
- 全部参数 `requires_grad=True`。

### `setup_model`（100–108）

`super().setup_model()` 后 `ClassificationModel.reshape_outputs(self.model, nc)` 对齐全连接输出维度。

### `build_dataset`（110–127）

`ClassificationDataset(..., augment=mode=="train", prefix=..., names=...)`。

### `get_dataloader`（129–160）

DDP 下 `torch_distributed_zero_first` 构建；无 `samples` 抛 `FileNotFoundError` 并附文档链接；`drop_last` 在 compile+train；`build_dataloader`；验证/测试模式把 `torch_transforms` 挂到 `model.transforms`（处理 parallel）。

### `preprocess_batch`（162–166）

`img`、`cls` 异步搬到 device。

### `progress_string`（168–176）

表头含 Epoch、GPU_mem、各 loss 名、Instances、Size。

### `get_validator`（178–182）

构造 `ClassificationValidator(test_loader, save_dir, args, callbacks)`。

### `plot_training_samples`（184–196）

加 `batch_idx`，`plot_images` 保存 `train_batch{ni}.jpg` 并 `join()` 等待绘图完成。