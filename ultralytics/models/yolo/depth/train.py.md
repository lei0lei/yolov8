# `ultralytics/models/yolo/depth/train.py`

`DepthTrainer` 继承 `DetectionTrainer`：换 `DepthModel`、深度 batch、深度验证器；`final_eval` 后在 checkpoint 写入深度标定。

## 逐段说明

### `__init__`（31–38）

`overrides["task"]="depth"` 后 `super().__init__`。

### `get_model`（40–47）

`DepthModel(ch, nc, ...)`，可选 `load(weights)`；无 `set_model_names_for_load` 检测专用路径。

### `preprocess_batch`（49–53）

`super().preprocess_batch` 后 `batch["depth"].float()`。

### `get_validator`（55–59）

`DepthValidator`。

### `plot_training_labels`（61–130）

重写检测版（无 bbox）：最多采样 1000 张 `_load_depth`；有效像素 `>0`；每图最多 cap 像素防 OOM；直方图+均值/中位数线；保存 `labels.jpg`，`on_plot` 回调。

### `final_eval`（132–176）

`super().final_eval()`；非 rank0 返回。
- rank0：`calibrate_checkpoint` 对 `best`/`last` 存在文件；`plot_dir` 在 plots 且为 plot_ckpt 时传 `save_dir`。
- 解析 `validation_split` 相对 `data["path"]`；写入 `depth_calibration` 到 trainer 属性。
- 异常仅 warning，checkpoint 保持未标定状态。