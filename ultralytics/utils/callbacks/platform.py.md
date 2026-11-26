# `ultralytics/utils/callbacks/platform.py`

Ultralytics **Platform** 云端训练流：`on_pretrain_routine_start` 发 `training_started` 并捕获控制台；`on_pretrain_routine_end` 应用远程取消；`on_fit_epoch_end` 异步上报 epoch 指标与可选 AVIF 训练图；`on_model_save` 限流上传 checkpoint；`on_train_end` 上传最佳权重、压缩曲线与 `training_complete`；`teardown` 停止控制台捕获。

## 逐段说明

### `slugify`（第 30–34 行）

- 空输入原样返回；小写、去非法字符、空格→连字符、合并连字符、截断 128。

### `_compact_plot`（第 37–55 行）

- **若** 含 `matrix`：列归一化，类维截断 100，附 `totals`。
- **否则**：类数>20 仅保留 mean 曲线与 AP；x 重采样 101 点插值 y；round 6 位小数。

### `_validation_payload`（第 58–74 行）

- 按 F1 排序图像指标；超 5000 条等间隔抽样；`extremes` 取最差/最好各 100。

### `_sanitize_json_value`（第 77–85 行）

- 递归 dict/list；非有限 float→None；非标 JSON 类型→str。

### `_send`（第 88–119 行）

- 无 `_api_key` 返回 None。
- `Platform` 客户端 `training.metrics` POST；401 清空 key。
- **分支**：`console_output` 事件在 key 已清空时不打 warning。
- `training_started` + 404 + model_id：返回 `{}` 供 resume 逻辑识别模型缺失。

### `_handle_control_response`（第 122–133 行）

- **若** `response.cancelled`：设 `ctx["cancelled"]`，`trainer.stop=True`。

### `_upload_file`（第 136–196 行）

1. 不存在则 warning；`models.upload_checkpoint` 取 URL。
2. visual：Content-Type AVIF，短超时；成功返回文件名。
3. 非 visual：`safe_upload` 600s；有 `gcsPath`+`run_id` 时发 `checkpoint_saved`。

### `_get_environment_info`（第 199–248 行）

- 汇总 ultralytics 版本、主机、OS、Python、CPU/RAM/磁盘、GIT 缓存字段。
- CUDA 可用时写 GPU 数量与类型；异常吞掉。

### `_get_project_name`（第 251–263 行）

- 本地路径形态则取最后一级有效 slug 段；`user/project` 形态 slugify 后半。

### `_get_training_gpus`（第 266–282 行）

- 非 cuda 返回 `[]`；解析 `device` 参数与 `CUDA_VISIBLE_DEVICES` 映射 NVML 索引；UUID 可见列表则返回 `None`（表示无法映射）。

### `_get_system_metrics`（第 285–295 行）

- 懒创建 `SystemLogger`（含训练 GPU 列表）；`get_metrics(rates=True)` 或 None。

### `on_pretrain_routine_start`（第 298–385 行）

1. pytest / 无 project / 无 API key / Python<3.11 → 早退。
2. 初始化 `trainer.platform` ctx（model/run/cancel/console/system 等）。
3. `ConsoleLogger` 批量 flush 到 `console_output` webhook（捕获在 training_started 成功后再 `start_capture`）。
4. 组 `training_started` payload：trainArgs、environment、system。
5. resume 读 `platform_model_id.txt`。
6. **若** model_id 对应 404 或 cancelled：重发无 model_id 的 started。
7. **若** 得 `modelId`：写记录、设 URL、开始捕获、`_handle_control_response`；**否则** `trainer.platform=None`。

### `on_pretrain_routine_end`（第 388–393 行）

- ctx 已 cancelled 则再次 `trainer.stop=True`（覆盖 `_setup_train` 重置）。

### `on_fit_epoch_end`（第 396–465 行）

- 无 ctx 则 return。
- 合并 metrics；epoch0 且 run_id 收集最多 3 张 train_batch 图。
- 附 optimizer lr、epoch0 modelInfo。
- 线程池 `_send_and_check_cancel`：可选 AVIF 上传 → `epoch_end` → 取消处理。

### `on_model_save`（第 468–487 行）

- 15 分钟限流；跳过进行中的上传；best 或 last → 异步 `_upload_file`。

### `on_train_end`（第 490–569 行）

1. 停控制台、等待 visual 上传。
2. 阻塞上传 best/last；收集 trainer/validator plots 按 type 去重；PR 曲线补 `classes`。
3. detect 任务附加 per-image validation 采样；depth 可选 calibration。
4. `training_complete` 高重试发送。

### `callbacks`（第 572–579 行）

- 六个键 + teardown 停捕获。