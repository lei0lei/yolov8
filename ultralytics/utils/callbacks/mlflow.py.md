# `ultralytics/utils/callbacks/mlflow.py`

MLflow 实验跟踪：在 **`on_pretrain_routine_end`** 设实验/运行并 `log_params`；**`on_train_epoch_end`** / **`on_fit_epoch_end`** 记训练与验证指标；**`on_train_end`** 上传工件并结束运行；**`teardown`** 在异常路径以 `FAILED` 结束。通过环境变量 `MLFLOW_*` 与 `trainer._mlflow_*` 标志控制行为。

## 逐段说明

### 导入与 MLflow 探测（第 23–34 行）

- 可选导入 `mlflow`，无包则 `mlflow=None`。

### `sanitize_dict`（第 37–39 行）

- 键去掉括号，值转 `float`，供 `log_metrics` 使用。

### `on_pretrain_routine_end`（第 42–83 行）

- **若** pytest 且当前测试名不含 `test_mlflow`：直接 return。
- `experiment_name` = 环境变量或 `project` 或 `/Shared/Ultralytics`；`run_name` = `MLFLOW_RUN` 或 `args.name`。
- 初始化 `trainer._mlflow_active/_mlflow_started_run` 为 False。
- **try**：`set_experiment`、`autolog`；无 active run 则 `start_run` 并置 `_mlflow_started_run`；`log_params`；`_mlflow_active=True`。
- **except**：warning，调用 `on_train_end(trainer, status="FAILED")`。

### `_log_metrics`（第 86–92 行）

- **try** `log_metrics(..., step=trainer.epoch)`。
- **except**：warning，`_mlflow_active=False`（不中断训练）。

### `on_train_epoch_end`（第 95–104 行）

- **若** `mlflow` 且 `_mlflow_active`：合并 sanitize 后的 lr 与 train loss → `_log_metrics`。

### `on_fit_epoch_end`（第 107–110 行）

- **若** 活跃：`_log_metrics` 验证 `trainer.metrics`。

### `on_train_end`（第 113–137 行）

- **若** 无 `mlflow`：return。
- **若** `status=="FINISHED"` 且活跃：**try** 记录 `weights` 父目录及 `save_dir` 下指定后缀文件。
- **若** `_mlflow_started_run`：根据 `MLFLOW_KEEP_RUN_ACTIVE` 决定是否 `end_run(status)`。

### `callbacks`（第 140–150 行）

- 含 `teardown` lambda 调用 `on_train_end(..., FAILED)`。