# `ultralytics/utils/events.py`

本模块实现可选的匿名使用分析：在满足 SETTINGS、进程 rank、在线状态、pip 或官方 git 源等条件时，将 train/val/predict/export 等运行元数据排队，经限速后 POST 到 Google Analytics Measurement Protocol。全局单例 `events` 与 `callbacks` 字典挂到训练/验证/预测/导出生命周期钩子。设计强调不干扰用户运行：无效 mode/task、队列满、速率未到期则早退；train/predict 分支内大量字段用 try/except 吞掉异常。

## 逐段说明

### `_post`（第 26～33 行）

- **第 28～31 行**：`json.dumps` 紧凑编码 POST body，设 `Content-Type: application/json`，`urlopen` 后 close。
- **第 32～33 行**：任意异常静默忽略。

### `_arch`（第 36～46 行）

- **第 42 行**：从 `model.description` 分词取导出模型架构名。
- **第 44 行**：从 `model.yaml` 或 `model.model.yaml` 取配置；SAM 等无 `.model` 时用 `{}`。
- **第 45 行**：`yaml_file` stem 或 description 第二 token 作为 stem，小写截断 100 字符。
- **第 46 行**：空则返回 `None`。

### 类 `Events`（第 49～202 行）

#### 类属性 `url`（第 69 行）

- GA4 collect 端点 URL（含 measurement_id 与 api_secret）。

#### `__init__`（第 71～93 行）

- **第 73～75 行**：初始化事件队列、30s 限速、`t=0` 上次发送时间。
- **第 76～86 行**：`metadata` 含 cli 是否 yolo、安装来源、python/torch 版本、CPU 信息、包版本、环境、`session_id` 随机、`engagement_time_msec`。
- **第 87～93 行**：`enabled` 需 `SETTINGS["sync"]`、`RANK in {-1,0}`、非测试、在线、且 pip 或 git origin 为 ultralytics 官方仓库。

#### `__call__`（第 95～202 行）— 门禁与入队

- **第 105～106 行**：未启用或 `cfg.mode`/`cfg.task` 不在 `MODES`/`TASKS` 则 return。
- **第 109 行**：队列长度 `< 25` 才追加新事件。
- **第 110～115 行**：`params` 合并 metadata、`task`、模型 basename（≤100）、`device` 字符串。
- **第 116～117 行**：`export` 模式附加 `format`。
- **第 118～153 行**：`train` 模式 try 块填充 data stem、imgsz、epochs_done、batch、hours、n、fitness、optimizer 名、lr0 最小 param_group、flags 逗号串、arch、ngpu；CUDA 时附加 GPU 名。
- **第 136～147 行**：`flags` 字典各训练相关布尔/条件项，真值键 join 为字符串。
- **第 152～153 行**：except 静默。
- **第 154～182 行**：`predict`/`track` 模式 try：n、pixels、speed 各阶段 ms、batch、format、nc、flags、arch、quantize、GPU、ONNX provider 等。
- **第 184～185 行**：去掉 None 值并最多保留 25 个键值，append `{"name": cfg.mode, "params": params}`。

#### `__call__` — 发送与重置

- **第 188～190 行**：未过 `rate_limit` 则 return。
- **第 193～198 行**：快照队列，守护线程调用 `_post`，payload 含 `SETTINGS["uuid"]` 与 events。
- **第 201～202 行**：清空队列并更新 `self.t`。

### 模块级对象与回调（第 205～237 行）

- **第 205 行**：`events = Events()` 单例。
- **第 208～211 行**：`on_train_end` 传入 trainer args/device/trainer。
- **第 213～219 行**：`on_val_start` 仅在 `validator.args.mode == "val"` 时上报，避免与 train 末次验证重复。
- **第 222～224 行**：`on_predict_end` 带 predictor。
- **第 227～229 行**：`on_export_start` 仅 args/device。
- **第 232～237 行**：`callbacks` 映射四个钩子名到上述函数。