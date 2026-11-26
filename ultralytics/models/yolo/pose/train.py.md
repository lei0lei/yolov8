# `ultralytics/models/yolo/pose/train.py`

`PoseTrainer` 继承 `DetectionTrainer`：`task=pose`，`PoseModel` 带 `kpt_shape`，校验 data 含关键点配置。

## 逐段说明

### `__init__`（42–56）

`overrides["task"]="pose"`。

### `get_model`（58–86）

`PoseModel(..., data_kpt_shape=...)` + load。

### `set_model_attributes`（88–97）

`super()` 后写 `kpt_shape`、`kpt_oks_sigmas`；无 `kpt_names` 时用数字串按类建默认 dict。

### `get_validator`（99–103）

`PoseValidator`。

### `get_dataset`（105–117）

`super().get_dataset()`；缺 `kpt_shape` 抛 `KeyError` 附 pose 数据集文档链接。