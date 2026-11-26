# `ultralytics/utils/callbacks/clearml.py`

ClearML 训练集成：在 **`on_pretrain_routine_start`** 初始化任务并连接超参；**`on_train_epoch_end`** 记录 Mosaic 与训练标量；**`on_fit_epoch_end`** 记录验证指标与模型信息；**`on_val_end`**（挂在 Validator）记录验证可视化；**`on_train_end`** 上传图表与最佳权重。`clearml` 不可用或 pytest 时 `callbacks` 为空字典。

## 逐段说明

### 条件导入（第 9–17 行）

- `TESTS_RUNNING` 为真则断言失败，不加载 ClearML。
- 成功则导入 `clearml`、`Task` 并校验 `__version__` 存在。
- **except**：`clearml = None`，后续不注册钩子。

### `_log_debug_samples`（第 20–38 行）

- **若** `Task.current_task()` 存在：遍历 `files`。
- 对每个 `f`：**若** `f.exists()`：用正则 `_batch(\d+)` 从文件名取 iteration；`report_image`，series 去掉 batch 段。

### `_log_plot`（第 41–58 行）

- 用 matplotlib 读图、无坐标轴显示；`report_matplotlib_figure`，`report_interactive=False`。

### `on_pretrain_routine_start`（第 61–87 行）

- **try**：
  - **若**已有 `Task.current_task()`：禁用 PyTorch/Matplotlib 自动绑定（`PatchPyTorchModelIO`、`PatchedMatplotlib` 的 `update_current_task(None)`）。
  - **否则** `Task.init`：项目名来自 `trainer.args.project`（去前导 `/`，空则 `Ultralytics`），`task_name=args.name`，`auto_connect_frameworks` 关 pytorch/matplotlib；打 warning 提示远程需先 `clearml-init`。
  - `task.connect(vars(trainer.args), name="General")`。
- **except**：warning，不记录本次运行。

### `on_train_epoch_end`（第 90–100 行）

- **若**当前 task 存在：
  - **若** `trainer.epoch == 1`：对 `train_batch*.jpg` 调 `_log_debug_samples`（标题 Mosaic）。
  - 对 `label_loss_items(tloss)` 与 `trainer.lr` 分别 `report_scalar`。

### `on_fit_epoch_end`（第 103–117 行）

- `report_scalar` Epoch Time。
- 遍历 `trainer.metrics`，按 `k` 的 `/` 前缀分 title。
- **若** `epoch == 0`：`model_info_for_loggers` → `report_single_value`。

### `on_val_end`（第 120–124 行）

- **若**有 task：对 `val*.jpg` 调 `_log_debug_samples`（Validation）。

### `on_train_end`（第 127–138 行）

- 合并 `trainer.plots` 与 `validator.plots` 键，**若**文件名不含 `"batch"` 则 `_log_plot`。
- `validator.metrics.results_dict` → `report_single_value`。
- `update_output_model(trainer.best)`。

### `callbacks` 字典（第 141–151 行）

- **若** `clearml` 非空则映射五个训练/验证钩子；否则 `{}`。