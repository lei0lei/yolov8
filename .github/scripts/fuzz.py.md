# `.github/scripts/fuzz.py`

该脚本对 `yolo` CLI 做蒙特卡洛模糊测试：随机/变异命令在隔离子进程执行，结果分为 pass、预期校验错误、环境跳过、网络 flake、超时、崩溃、bug 候选；对候选二次确认后写 JSONL 与 findings。`report` 子命令用 stdlib 聚合分片结果并按签名去重开 GitHub issue。`repro` 重放单条命令。

## 逐段说明

### 模块级常量（第 37–236 行）

超时缩放、`MODE_TIMEOUTS`、`PERSONALITIES` 权重、`NEVER_MUTATE`/`CLAMPS`、`EXPORT_POOL`、`ALTERNATE_CORPUS`、`SAFE_BOUNDARIES`、`ENUM_POOLS`/`PROBES`、`DATASET_MUTATIONS`、`COMBO_POOL`、`EXPECTED_MODULES`/`NETWORK_MARKERS` 等分类 oracle。

### `load_universe`（第 239–265 行）

延迟导入 `TASKS`、CFG 类型键、默认 cfg、`ASSETS` 源路径与可选 coreml。

### `precache_assets`（第 268–283 行）

下载各任务权重与数据；调用 `prepare_sources/datasets/models`。

### `prepare_models`（第 286–309 行）

在 `fuzz-models` 写入截断/空/随机/onnx 垃圾/图片冒充 pt 等无效权重路径列表。

### `prepare_datasets` / `make_dataset`（第 312–368 行）

按 `DATASET_MUTATIONS` 生成合成 detect 数据集 yaml（各类坏标签/缺 val/坏 yaml 等）。

### `prepare_sources`（第 371–405 行）

有效/无效图片、unicode 路径、短视频 URL 列表供 predict/track。

### `strip_defaults` / `build_corpus`（第 408–439 行）

种子语料：每任务×模式标准命令 + RT-DETR/YOLOE 变体。

### `sample_trial`（第 442–524 行）

按 personality 选策略（invalid/combo/malformed/model/dataset/source）；变异 argv 并记录 `valid_input`。

### `probe_supported` / `sample_mutation` / `malformed_tokens`（第 527–579 行）

枚举/数值/布尔探针与畸形 token 生成。

### `run_trial`（第 582–628 行）

临时 cwd、export 复制权重、进程组超时杀树、返回 rc/stderr/耗时。

### `parse_traceback` / `make_signature` / `classify`（第 631–711 行）

解析 ultralytics 栈帧；分类 pass/timeout/crash/env-skip/flake/expected/bug-candidate（含 T1/T2 语义）。

### `stderr_tail` / `command_hash` / `read_history` / `write_history`（第 714–744 行）

日志尾部、命令哈希、7 天探索历史读写。

### `cmd_fuzz`（第 747–876 行）

**第一段**：环境信息与 canary 跑 corpus。**第二段**：主循环采样、去重、磁盘门槛、`execute` 含 flake 重试与 hang 确认。**第三段**：写 findings JSON 与计数摘要。

### `cmd_repro`（第 879–924 行）

便携路径 remap；多次 `run_trial`；非 pass/expected 一致则 exit 1。

### `gh` / `cmd_report`（第 927–1108 行）

聚合 findings；T1 单独开 issue；T2 滚入 umbrella；关闭 issue 回归评论；Actions summary。

### `issue_body` / `main`（第 1111–1173 行）

issue 模板；子命令 fuzz/repro/report 分发。