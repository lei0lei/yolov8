# `ultralytics/engine/tuner.py`

本模块实现 YOLO **超参数演化（Evolutionary Tuning）**：在预设搜索空间内对 lr、增强、损失增益等变异，用短训 fitness 选优并迭代。核心类 `Tuner`；对外由 `YOLO.tune()` / `model.tune(...)` 构造并调用 `__call__`。每次迭代用本仓库 `MultiTrainer` 触发真实训练（可走 CLI 子进程），结果写入 `tune_results.ndjson`，可选 MongoDB Atlas 做多机共享。与 `BaseTrainer` 的关系：不继承 Trainer，只编排多次 `train()` 并读回 `TASK2METRIC` 与 `fitness`。

---

## 导入说明（第 15–27 行）

| 组别 | 内容 | 用途 |
|------|------|------|
| 标准库 | `json`、`shutil`、`time`、`datetime` | NDJSON、清理目录、耗时、Mongo 时间戳 |
| 数值 | `numpy` | 历史矩阵、变异采样 |
| 配置 | `CFG_INT_KEYS`、`TASK2METRIC`、`get_cfg`、`get_save_dir` | 整型超参、任务主指标、保存路径 |
| 工具 | `LOGGER`、`YAML`、`callbacks`、`colorstr`、`remove_colorstr` | 日志、写出最佳 yaml、回调 |
| 可选依赖 | `check_requirements` | Mongo 前检查 pymongo |
| 绘图 | `plot_tune_results` | 每轮更新调参曲线图 |

模块 docstring（第 2–13 行）给出 `model.tune(...)` 的典型用法示例。

---

## 逐段说明

### 类 `Tuner`（第 30–619 行）

持有搜索空间、基础权重、本地/Mongo 存储与变异逻辑；可调用对象执行多轮 train+记录。

---

### `Tuner.__init__`（第 79–142 行）

1. **`space`**：`args.pop("space")` 或默认大字典，每项为 `(min, max)` 或 `(min, max, gain)`。
2. **Mongo 参数**：从 `args` pop `mongodb_uri/db/collection`。
3. **`get_cfg(overrides=args)`** → `self.args`；无 `model` 则 `YOLO(self.args.model).model` 作为每轮起点权重。
4. **目录**：`exist_ok` 与 resume 对齐；`tune_dir = get_save_dir(..., name or "tune")`；随后把 `name/exist_ok/resume` 清零以免污染子训练。
5. **`tune_file`**：`tune_dir/tune_results.ndjson`；默认 callbacks + `add_integration_callbacks`。
6. 有 URI 则 **`_init_mongodb`**；打日志与文档链接。

**结果**：`Tuner` 实例就绪，尚未训练；`self.model` 为共享初始 `nn.Module`。

---

### `Tuner._connect`（第 144–186 行）

1. `check_requirements("pymongo")`；导入 `MongoClient` 与连接异常类。
2. 循环最多 `max_retries`：建 client（超时/池化/retry 配置）→ `admin.ping`。
3. 成功则日志并返回 client；`ConnectionFailure`/`ServerSelectionTimeoutError` 时指数退避 `2**attempt` 秒再试；末次失败 re-raise。

---

### `Tuner._init_mongodb`（第 188–206 行）

`self.mongodb = _connect(uri)`；`self.collection = db[collection]`；日志启用分布式调参。注释说明新集合建 fitness 索引、连接失败上抛。

---

### `Tuner._json_default`（第 208–211 行，静态）

序列化 NDJSON 时：有 `.item()` 的张量标量化，否则 `str(x)`。

---

### `Tuner._result_record`（第 213–233 行）

组装单条记录 dict：`iteration`、`fitness`（5 位小数）、`hyperparameters`、`datasets`；可选 `save_dirs`、`failed_datasets`。

---

### `Tuner._save_to_mongodb`（第 235–270 行）

`insert_one`：fitness、超参（item 化）、metrics、datasets、save_dirs、failed、本地时区 timestamp；**iteration** 用 `find_one_and_update` 递增 `defaults.last_iteration` 文档计数。

异常仅 `LOGGER.warning`，不中断调参。

---

### `Tuner._sync_mongodb_to_file`（第 272–303 行）

1. `find({fitness exists}).sort(iteration)` 拉全表。
2. 空则返回；否则 `update_one` defaults 的 `last_iteration` 为 max。
3. **重写**本地 `tune_file` 每行一条 JSON（经 `_result_record`）。

目的：分布式下本地 NDJSON 与 Mongo 一致，resume/变异/绘图同源。

---

### `Tuner._load_local_results`（第 305–310 行）

无文件 → `[]`；否则逐行 `json.loads`。

---

### `Tuner._local_results_to_array`（第 312–323 行）

行 → 矩阵：列0 fitness，后续列按 `self.space` 键序取 `hyperparameters[k]` 或 `getattr(self.args,k)`；无历史 → `None`。

---

### `Tuner._save_local_result`（第 325–328 行）

追加模式写一行 NDJSON。

---

### `Tuner._best_metrics`（第 330–338 行，静态）

单 dataset → 其 metrics dict；多 dataset → 各 dataset 的 fitness 四舍五入；无则 `None`。

---

### `Tuner._has_training_metrics`（第 340–346 行，静态）

`datasets` 中非 failed 且 metrics 非空；`require_all` 控制 all/any。

---

### `Tuner._best_result_index`（第 348–352 行，类方法）

优先在“有训练指标”的行中取 fitness 最大索引；否则全局 argmax。

---

### `Tuner._mutate`（第 354–458 行）

**核心变异**；返回下一组 `dict[str, float]` 超参。

#### 段 1：历史来源 Mongo（373–396）

- 有 Mongo 且存在带 fitness 文档：拼 `history` 数组 `[fitness, h1, h2, ...]`。
- 否则尝试 `insert_one({_id:defaults})`；`DuplicateKeyError` 表示别 worker 已初始化 → 用默认 hyp 一行 history；建 `fitness` 降序索引；`history is None` 时直接返回 **`_constrain(default_hyp)`**。

#### 段 2：回退本地 NDJSON（398–401）

`history is None` → `_load_local_results` + `_local_results_to_array`。

#### 段 3：有历史的进化变异（404–454）

1. `fitness` 四舍五入；`stale` 距上次最优代数；取 top `n` 父代 `x`；`bounds`/`span`/`mutable`（span>0）。
2. 归一化基因 `population`、fitness 权重、每维 `gains`、整数键 `resolution`；`decay = 1 - 0.2*min(stale/25,1)` → `scale`。
3. 历史≥30：精英协方差 `multivariate_normal` 探索（40%*confidence）或高斯扰动；200 次尝试避免与 `existing` 元组重复。
4. 反归一化得 `hyp`，**`_constrain`**，唯一则返回。
5. 200 次仍重复 → `RuntimeError`；搜索空间无可变维 → exhausted `RuntimeError`。

#### 段 4：无历史（455–458）

`hyp = {k: getattr(self.args,k) for k in space}` → **`_constrain`** 返回。

---

### `Tuner._constrain`（第 460–471 行）

遍历 `space`：`CFG_INT_KEYS` → ceil/floor 边界内 round int；浮点 round 5 位并 clamp 到 bounds。

---

### `Tuner.__call__`（第 473–619 行）

**主循环**；参数 `iterations`（默认 300）、`cleanup`（删非最优 run 目录）。

#### 段 1：准备与 resume（487–502）

- 懒导入 `MultiTrainer`；创建 `tune_dir/weights`；`best_save_dirs`、成功计数。
- Mongo 启动 **`_sync_mongodb_to_file`**。
- 已有 NDJSON → `start = len(results)`，从 `start+1` 继续。

#### 段 2：单次迭代 train（503–544）

1. **`mutated_hyp = _mutate()`**。
2. `train_args = vars(args) ∪ mutated_hyp`；`data` pop 并规范为 list。
3. **`MultiTrainer(None, {**train_args, data}, self.model).train()`** — `trainer=None` 走 **CLI 子进程** `yolo train`；基座权重来自 `self.model`。
4. 汇总 `failed_datasets`、`dataset_metrics`（失败填 0）、`mean_metrics`；**fitness** = 各 dataset fitness 均值。
5. **`_result_record`**；`require_all` 统计成功轮数。
6. Mongo：`_save_to_mongodb` + sync；达全局 `iterations` 文档数设 `stop_after_iteration`；否则 **`_save_local_result`**。

#### 段 3：最优追踪与磁盘清理（546–571）

- 重载 NDJSON → `x` 数组；**`_best_result_index`**；若当前即最优且 `cleanup`：删旧 best 目录、把本轮 `weights/*.pt` 拷到 `tune_dir/weights`（多 dataset 分子目录）；非最优且 cleanup → `shutil.rmtree` 本轮 save_dir。

#### 段 4：绘图、日志与最佳 yaml（573–619）

- **`plot_tune_results(tune_file)`**。
- 按成功比例打 status；有有效 best 则日志 fitness、metrics、权重路径说明。
- 有训练指标 → 从 `x[best_idx,1:]` 写 **`best_hyperparameters.yaml`**（整型键转 int）并 `YAML.print`；全无指标则 error 跳过。
- Mongo 达目标迭代 → `break`。

**与 Trainer 训练要素**：每轮完整 `train()`（含 epoch/batch/AMP/EMA/DDP/断点），但 epoch 数通常很小；Tuner 不直接操控这些，只改 `train_args` 超参。

---

## 方法覆盖清单

| 方法 | 行号 |
|------|------|
| `__init__` | 79–142 |
| `_connect` | 144–186 |
| `_init_mongodb` | 188–206 |
| `_json_default` | 208–211 |
| `_result_record` | 213–233 |
| `_save_to_mongodb` | 235–270 |
| `_sync_mongodb_to_file` | 272–303 |
| `_load_local_results` | 305–310 |
| `_local_results_to_array` | 312–323 |
| `_save_local_result` | 325–328 |
| `_best_metrics` | 330–338 |
| `_has_training_metrics` | 340–346 |
| `_best_result_index` | 348–352 |
| `_mutate` | 354–458 |
| `_constrain` | 460–471 |
| `__call__` | 473–619 |

调用链摘要：`YOLO.tune` → `Tuner(...)` → `Tuner()` → 循环 `{_mutate → MultiTrainer.train → 记录/清理 → plot → yaml}`。