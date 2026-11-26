# `ultralytics/utils/callbacks/raytune.py`

Ray Tune 超参搜索集成：仅在 **`on_fit_epoch_end`** 向当前 trial 上报 `trainer.metrics` 与 `epoch+1`。要求 Ray≥2.41（`tune.get_context`）；不可用则 `callbacks` 为空。

## 逐段说明

### 条件导入（第 3–8 行）

- 导入 `ray.tune`，断言 `get_context` 存在；失败则 `tune=None`。

### `on_fit_epoch_end`（第 11–28 行）

- **若** `tune.get_context().get_trial_id()` 非空（处于活跃 Tune session）：`tune.report({**trainer.metrics, "epoch": trainer.epoch + 1})`。
- **否则** 无操作。

### `callbacks`（第 31–37 行）

- **若** `tune` 可用，仅注册 `on_fit_epoch_end`。