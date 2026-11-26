# `ultralytics/models/rtdetr/train.py`

`RTDETRTrainer` 继承检测 `DetectionTrainer`，复写模型构建、数据集与验证器工厂。类属性 `get_class_counts` 指回 `BaseTrainer.get_class_counts`，因 RT-DETR 损失不使用类别权重。数据集使用 `RTDETRDataset`，`rect=False` 固定；验证器为 `RTDETRValidator` 拷贝 args。

## 逐段说明

### 行 48：`get_class_counts`

绑定基类方法，跳过 DetectionTrainer 中供分类加权的计数逻辑。

### `get_model(self, cfg=None, weights=None, verbose=True)`（行 50）

- **行 61–63**：`set_model_names_for_load(RTDETRDetectionModel(cfg, nc, ch, verbose=...))` 构建网络。
- **行 64–65**：有 `weights` 则 `model.load(weights)`。
- **行 66**：返回 `model`。

### `build_dataset(self, img_path, mode="train", batch=None)`（行 68）

构造 `RTDETRDataset`：
- `augment = (mode == "train")`；
- `rect=False`；
- `fraction`：数据 `complete` 时为 1.0，否则 `get_split_fraction` 按 train/val split；
- 传入 `hyp`、`tile`、`cache`、`classes`、`data` 等来自 `self.args`/`self.data`。

### `get_validator(self)`（行 97）

返回 `RTDETRValidator(self.test_loader, save_dir, args=copy(self.args), _callbacks=self.callbacks)`。