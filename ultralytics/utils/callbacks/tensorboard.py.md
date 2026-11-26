# `ultralytics/utils/callbacks/tensorboard.py`

TensorBoard 标量与计算图：`on_pretrain_routine_start` 创建 `SummaryWriter`；`on_train_start` 尝试 JIT 追踪写图；`on_train_epoch_end` / `on_fit_epoch_end` 写 loss、lr、验证指标；`teardown` 关闭 writer。pytest 或 SummaryWriter 不可用时无钩子。

## 逐段说明

### 条件导入（第 7–20 行）

- 非测试、懒加载 `SummaryWriter`；失败则 `SummaryWriter=None`。

### `_log_scalars`（第 23–38 行）

- **若** 全局 `WRITER` 存在：对字典每项 `add_scalar(k, v, step)`。

### `_log_tensorboard_graph`（第 41–85 行）

- 由 `imgsz`、`channels` 构造全零 dummy 输入；`model.eval()`。
- **分支 1（YOLO）**：`unwrap_model` + `jit._trace.trace` → `add_graph`；成功 return。
- **分支 2（RT-DETR 等）**：deepcopy、fuse、对带 `export` 的模块设 `export=True, format=torchscript`、dry run 再 trace。
- 两次失败则 warning 链式异常信息。

### `on_pretrain_routine_start`（第 88–96 行）

- **若** SummaryWriter 可用：**try** 设 `WRITER=SummaryWriter(save_dir)`；**except** warning。

### `on_train_start`（第 99–102 行）

- **若** `WRITER`：调用 `_log_tensorboard_graph`。

### `on_train_epoch_end`（第 105–108 行）

- `_log_scalars` train loss（`epoch+1` 为 step）与 `trainer.lr`。

### `on_fit_epoch_end`（第 111–113 行）

- `_log_scalars(trainer.metrics, epoch+1)`。

### `teardown`（第 116–121 行）

- **若** `WRITER`：`close()` 并置 `None`（含失败训练路径）。

### `callbacks`（第 124–134 行）

- SummaryWriter 可用时注册五个键。