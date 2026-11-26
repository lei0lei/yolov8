# `ultralytics/utils/callbacks/dvc.py`

DVCLive 集成：预训练初始化；训练参数与指标、学习率；按时间戳去重上传训练/验证图；训练结束记录混淆矩阵与 `best` 工件。钩子：`on_pretrain_routine_start/end`、`on_train_start`、`on_train_epoch_start`、`on_fit_epoch_end`、`on_train_end`。需 `dvclive>=2.11.0` 且非 pytest。

## 逐段说明

### 条件导入与全局状态（第 7–25 行）

- 校验非测试环境、`dvclive` 版本、`live=None`、`_processed_plots={}`、`_training_epoch=False`。
- 失败则 `dvclive=None`。

### `_log_images`（第 28–52 行）

- **若** `live` 存在：解析 `_batch(\d+)`，将路径重排为「stem/批次号」以支持 UI 滑条；`live.log_image`。

### `_log_plots`（第 55–66 行）

- 比较 `plots` 中 timestamp 与 `_processed_plots`；变化则 `_log_images` 并更新缓存。

### `_log_confusion_matrix`（第 69–89 行）

- **若** `validator.confusion_matrix` 为 None（如 depth）则 return。
- 将矩阵展开为 targets/preds 列表；`live.log_sklearn_plot` 归一化混淆矩阵。

### `on_pretrain_routine_start`（第 92–99 行）

- 创建 `dvclive.Live(save_dvc_exp=True, cache_images=True)`；失败 warning。

### `on_pretrain_routine_end`（第 102–104 行）

- `_log_plots(trainer.plots, "train")`。

### `on_train_start`（第 107–110 行）

- **若** `live`：`log_params(trainer.args)`。

### `on_train_epoch_start`（第 113–116 行）

- `_training_epoch = True`。

### `on_fit_epoch_end`（第 119–149 行）

- **仅当** `live` 且 `_training_epoch`：
  - 合并 train loss、metrics、lr → `log_metric`。
  - **epoch==0**：`model_info_for_loggers`，`plot=False`。
  - `_log_plots` train 与 val。
  - `live.next_step()`；`_training_epoch=False`（区分最终最佳模型评估与常规 fit epoch）。

### `on_train_end`（第 152–180 行）

- **若** `live`：再记一次指标（`plot=False`）；val 图；混淆矩阵；**若** `trainer.best.exists()` 则 `log_artifact`；`live.end()`。

### `callbacks`（第 183–194 行）

- `dvclive` 可用时注册六个钩子。