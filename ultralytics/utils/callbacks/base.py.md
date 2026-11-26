# `ultralytics/utils/callbacks/base.py`

本文件定义 Ultralytics **全套生命周期钩子**的占位实现与默认注册表，覆盖 **Trainer**（预训练、训练、保存、结束、拆卸）、**Validator**（验证起止与 batch）、**Predictor**（预测与后处理）、**Exporter**（导出起止）。`add_integration_callbacks` 负责按实例类型与 `SETTINGS` 注入 `events`、Platform 及各训练日志集成。

## 逐段说明

### Trainer 占位回调（第 11–71 行）

以下函数体均为空，仅作文档化占位；引擎在对应事件点按名调用 `instance.callbacks[event]` 中的可调用对象：

| 行号 | 函数 | 触发时机 |
|------|------|----------|
| 11–12 | `on_pretrain_routine_start` | 预训练例程开始（数据加载与模型装配之前） |
| 15–16 | `on_pretrain_routine_end` | 预训练例程结束 |
| 19–20 | `on_train_start` | 训练开始、第 0 个 epoch 之前 |
| 23–24 | `on_train_epoch_start` | 每个训练 epoch 开始 |
| 27–28 | `on_train_batch_start` | 每个训练 batch 前向之前 |
| 31–32 | `optimizer_step` | 优化器步进（**默认不调用**，预留集成） |
| 36–37 | `on_before_zero_grad` | 梯度清零之前（**默认不调用**） |
| 39–43 | `on_train_batch_end` | 每个训练 batch 反传之后（含梯度累积延迟步进） |
| 46–47 | `on_train_epoch_end` | 训练 epoch 结束、验证之前 |
| 50–54 | `on_fit_epoch_end` | 训练+验证整轮结束；训练结束后对最佳模型再调一次 |
| 57–58 | `on_model_save` | 验证后保存 checkpoint |
| 61–62 | `on_train_end` | 训练结束、最终评估之后 |
| 65–66 | `on_params_update` | 参数更新（**默认不调用**） |
| 69–70 | `teardown` | 训练拆卸 |

### Validator 占位回调（第 76–89 行）

- `on_val_start` / `on_val_batch_start` / `on_val_batch_end` / `on_val_end`：验证阶段起止与 batch 级钩子。

### Predictor 占位回调（第 95–112 行）

- `on_predict_start` 至 `on_predict_end`，含 `on_predict_postprocess_end`（后处理结束）。

### Exporter 占位回调（第 118–123 行）

- `on_export_start` / `on_export_end`：模型导出起止。

### `default_callbacks`（第 126–156 行）

- 字典键为事件名，值为**单元素列表**，每个元素为上述同名占位函数。
- 注释标明事件所属引擎（trainer / validator / predictor / exporter）。

### `get_default_callbacks`（第 159–172 行）

- 返回 `defaultdict(list, deepcopy(default_callbacks))`。
- **分支**：使用 `list` 作为 default_factory，使后续 `append` 集成回调时各事件键自动获得可扩展列表；`deepcopy` 避免多实例共享同一列表引用。

### `add_integration_callbacks`（第 175–210 行）

1. 导入 `SETTINGS` 与 `events_cb`，初始列表 `callbacks_list = [events_cb]`。
2. **若** `instance.__class__.__name__` 含 `"Trainer"`：
   - 追加 `platform_cb`。
   - 对键 `clearml, comet, dvc, mlflow, raytune, tensorboard, wandb` 遍历：`SETTINGS[key] is True` 时动态 `import_module`，`wandb` 映射到子模块 `wb`，取其 `callbacks` 字典追加。
3. 对每个集成的 `callbacks` 字典：对每个 `(k, v)`，**若** `v` 尚不在 `instance.callbacks[k]` 中则 `append`（去重，不覆盖用户已注册的同函数引用）。