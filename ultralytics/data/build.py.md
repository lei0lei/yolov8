# `ultralytics/data/build.py`

本模块是 **训练 DataLoader 工厂** 与 **预测输入源路由** 的集中点：无限循环 `InfiniteDataLoader`、分布式 **连续 batch 采样** `ContiguousDistributedSampler`、按任务实例化 `dataset.py` 中各类 Dataset，以及 `check_source` / `load_inference_source` 把字符串、摄像头、流、张量等映射到 `loaders.py` 中的迭代器。`engine.trainer` 调用 `build_yolo_dataset` + `build_dataloader`；`engine.predictor` 经 `load_inference_source` 消费推理源。`cfg/default.yaml` 中的 `cache`、`rect`、`tile`、`fraction` 等经 `build_yolo_dataset` 传入数据集构造函数。

**图像尺寸**：`cfg.imgsz` 传给 Dataset；depth 验证用 `LetterBox(scale_fill=True)` 拉伸而非 pad。**Rect 训练**：`pad` 训练为 0、验证为 0.5，与 `set_rectangle` 中 `batch_shapes` 对齐。

## 逐段说明

### 导入（第 3–42 行）

- **分布式与数据**：`torch.distributed`、`Dataset`、`dataloader`、`distributed`。
- **`dataset`**：各任务 Dataset 类（延迟避免循环依赖的只有类型使用处）。
- **`loaders`**：`LOADERS` 元组、`Load*` 类、`SourceTypes`、`autocast_list`。
- **`utils`**：`IMG_FORMATS`/`VID_FORMATS`、`add_polygon_background`、`get_split_fraction`。
- **检查与设备**：`check_file`、`get_torch_device_backend`、`TORCH_*` 版本门控。

---

### `InfiniteDataLoader`（第 45–107 行）

继承 `DataLoader`，用 `_RepeatSampler` 包裹 `batch_sampler`，使 **epoch 结束后 worker 不销毁**，`__iter__` 仍只产出 `len(self)` 个 batch（一轮），但底层 sampler 无限重复。

#### `__init__`（第 70–76 行）

`TORCH_2_0` 为假时去掉 `prefetch_factor`；`object.__setattr__` 替换 `batch_sampler`（因 PyTorch 可能只读）；`iterator=None` 推迟 fork worker。

#### `__len__`（第 78–80 行）

返回底层 **样本 sampler 长度**（非重复次数）。

#### `__iter__`（第 82–87 行）

首次 `_get_iterator()`；循环 `len(self)` 次 `next(iterator)` — 一轮 batch 数。

#### `__del__`（第 89–97 行）

强杀仍存活 worker 线程并 `close()`，吞异常避免析构报错。

#### `close`（第 99–102 行）

调用 iterator `_shutdown_workers`，释放管道与 SIGCHLD 注册。

#### `reset`（第 104–107 行）

`close` + `iterator=None`，用于训练中改数据集后重建迭代器。

---

### `_RepeatSampler`（第 110–127 行）

#### `__init__`（第 120–122 行）

保存被包装的 `sampler`。

#### `__iter__`（第 124–127 行）

`while True: yield from iter(sampler)` — DataLoader 多次 `next` 时跨 epoch 复用同一 worker 顺序。

---

### `ContiguousDistributedSampler`（第 130–232 行）

DDP 验证 **rect/tile 按尺寸分组** 时，按 **连续 batch 块** 切分到各 rank，而非默认 round-robin。余数 batch 分配给前几个 rank。

#### `__init__`（第 162–186 行）

默认 `world_size`/`rank` 来自 `dist`；`batch_size` 默认 `dataset.batch_size`；`total_size=len(dataset)`；`source_ends` 来自 `dataset.source_offsets`（tile 验证对齐源边界）；若 `batch_size >= total_size` 则降为 1；`num_batches = ceil(total/batch_size)`。

#### `_get_rank_indices`（第 188–207 行）

- `batches_per_rank_base = num_batches // num_replicas`，`remainder = num_batches % num_replicas`。
- 本 rank 批次数 = base + (rank < remainder)。
- `start_batch = rank*base + min(rank, remainder)`，`end_batch = start + batches_for_this_rank`。
- 样本索引 `[start_batch*bs, end_batch*bs)` 钳到 `total_size`。
- 若有 `source_ends`：用 `searchsorted` 将起止索引 **扩到完整源组**（不把一张大图 tile 拆到两卡）。

#### `__iter__`（第 209–219 行）

`range(start,end)`；`shuffle` 时 `Generator.manual_seed(epoch)` + `randperm` 确定性洗牌。

#### `__len__` / `set_epoch`（第 221–232 行）

返回本 rank 样本数；`set_epoch` 更新洗牌种子。

---

### `seed_worker`（第 235–239 行）

`worker_id` 进程内：`torch.initial_seed() % 2**32` 同步设 `numpy`/`random`，与主进程 `generator` 配合可复现增广。

---

### `build_yolo_dataset`（第 242–306 行）

按 **`cfg.task`** 与 `data` YAML 选类并构造：

| 分支 | Dataset 类 | pad/rect 特殊 |
|------|------------|---------------|
| `depth` | `DepthDataset` | val 忽略 rect pad |
| `semantic` | 有 `masks_dir` → `SemanticDataset`，否则 `PolygonSemanticDataset`（先 `add_polygon_background`） | pad=0 |
| `multi_modal` | `YOLOMultiModalDataset` | — |
| 默认 | `YOLODataset` | train pad=0，else 0.5 |

`data.complete` 时 `fraction=1.0`；否则 `get_split_fraction(cfg.fraction, train/val)`。统一传入 `imgsz`、`cache`、`tile`、`classes`、`stride` 等。

---

### `build_grounding`（第 309–351 行）

直接构造 `GroundingDataset`，`json_file`、`max_samples`；其余超参与 YOLO 类似（`pad` 训练 0 / 验证 0.5）。

---

### `build_dataloader`（第 354–422 行）

1. **`batch = min(batch, dataset_len)`** — 小数据集不越界。
2. **`seed = initial_seed() - RANK - 1`** — xdist 各 rank 区分。
3. **Sampler**：单卡 `sampler=None`；DDP 且 `shuffle` → `DistributedSampler`；DDP 不 shuffle → `ContiguousDistributedSampler`。
4. **`drop_last`**：仅当 `samples > batch > 0` 且不能整除时生效。
5. **`nw`（workers）**：`min(cpu_count//max(min(nd,WORLD_SIZE),1), workers, 0若batches<=1 else batches)` — 避免单 batch 仍起进程池；CPU/MPS `nd=0`。
6. **`generator`** 固定大素数种子模 `2**64`。
7. **`pin_memory`**：有 GPU 类设备且 `pin_memory`；NPU/XPU 且版本允许时设 `pin_memory_device`。
8. 返回 `InfiniteDataLoader(..., collate_fn=dataset.collate_fn, prefetch_factor=4训练/2验证)`。

---

### `check_source`（第 425–483 行）

返回六元组：`(source, webcam, screenshot, from_img, in_memory, tensor)`。

**字符串/整数/Path**：

- RTSP/RTMP/TCP 或数字摄像头或 `.streams` 或（HTTP 且非静态图文件）→ **`webcam=True`**。
- `screen` / `screen …` 且路径不存在 → **`screenshot`**。
- URL 带图/视频后缀 → `check_file` 下载。
- 纯 URL：HEAD/GET 3s，`Content-Type` 非 `image/*` 则流；否则 `autocast_list` + `from_img`。

**已是 `LOADERS` 实例** → `in_memory`。**list/tuple** → `autocast_list`。**PIL/ndarray** → `from_img`。**Tensor** → `tensor`。否则 `TypeError`。

---

### `load_inference_source`（第 486–533 行）

1. `check_source` 得标志位；`SourceTypes(stream, screenshot, from_img, tensor)` 挂到 dataset。
2. 分支：`LoadTensor` / 已有 loader / `LoadStreams` / `LoadScreenshots` / `LoadPilAndNumpy` / 默认 `LoadImagesAndVideos`。
3. 设置 **`dataset.source_type`** 供 Predictor 后处理分支使用。