# `ultralytics/utils/callbacks/wb.py`

Weights & Biases 集成：`on_pretrain_routine_end` 初始化/恢复 run；`on_train_epoch_end` 训练 loss 与 lr；`on_fit_epoch_end` 图、首 epoch 模型信息、验证指标；`on_train_end` 工件与 PR 曲线。非 pytest 且已安装 `wandb` 时生效。

## 逐段说明

### 条件导入（第 6–14 行）

- 校验非测试、`wandb` 包；`_processed_plots` 防重复。

### `_custom_table`（第 17–48 行）

- Polars 构造 class/x/y 表，round 数值；`wb.plot_table` 使用 `wandb/area-under-curve/v0` 模板。

### `_plot_curve`（第 51–93 行）

- 在 x 轴上插值 `num_x` 点；先记 mean 曲线再逐类扩展 `x_log/y_log/classes`；`wb.log(..., commit=False)`。

### `_log_plots`（第 96–115 行）

- 浅拷贝 `plots`；timestamp 变化则 `wb.Image` 以 `name.stem` 为键记录。

### `on_pretrain_routine_end`（第 118–147 行）

- **若** 尚无 `wb.run`：
  - 从 `save_dir/wandb` 找最新 `run-*.wandb` 用于 resume ID。
  - 动态添加 `on_train_end`/`teardown` 的 `wb.finish`（teardown 用 `exit_code=1`）。
  - `wb.init`：项目名清洗与截断 128；`config=args`；`resume="allow"` 或新 ID（名称+时间戳）。
  - `define_metric` 为 train/lr/metrics/val 前缀绑定 `step_metric=epoch`。
- 始终 `config.setdefaults(vars(trainer.args))`。

### `on_fit_epoch_end`（第 150–156 行）

- `_log_plots` train 与 val；**epoch==0** 记 `model_info_for_loggers`；`log` metrics + epoch，`commit=True`。

### `on_train_epoch_end`（第 159–162 行）

- train loss 与 lr，`commit=False`。

### `on_train_end`（第 165–186 行）

- 再次 `_log_plots`；**若** save 且 best 存在：Artifact 上传 best。
- **若** plots 且 metrics 有 curves：按 `ap_class_index` 逐曲线 `_plot_curve`。
- 最后 `log epoch`，`commit=True`。

### `callbacks`（第 189–198 行）

- 四个训练钩子。