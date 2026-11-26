# `ultralytics/engine/trainer.py`

本模块是 Ultralytics YOLO 训练的**引擎核心**：`BaseTrainer` 封装单任务训练的完整生命周期（配置解析、数据集、模型装配、DDP 子进程、epoch/batch 循环、AMP、EMA、验证、断点与 metrics），各任务子类（如 `DetectionTrainer`）只覆写数据加载、模型构建与验证器钩子。对外主入口是 `train()`；`YOLO(...).train()`、`yolo train` CLI 经 `Model` 层实例化具体 Trainer 并调用。多数据集串联微调由同文件的 `MultiTrainer` 完成，并被 `Tuner` 用于超参搜索。模块依赖 `cfg`、`data`、`utils/torch_utils`、`utils/dist` 及任务侧 `nn/tasks`。

---

## 导入说明（第 9–78 行）

| 组别 | 内容 | 用途 |
|------|------|------|
| 标准库 | `gc`、`math`、`os`、`subprocess`、`time`、`warnings`、`contextlib.nullcontext`、`copy`、`datetime`、`functools.partial`、`Path` | 进程/内存/时间、DDP 子进程、上下文与路径 |
| 数值与张量 | `numpy`、`torch`、`torch.distributed`、`nn`、`optim` | 训练计算与分布式 |
| 配置与数据 | `get_cfg`、`get_save_dir`、数据集检查与 NDJSON 转换 | 参数合并、保存目录、数据元信息 |
| 模型 | `DistillationModel`、`load_checkpoint`、`MuSGD` | 蒸馏、权重加载、自定义优化器 |
| 工具常量与回调 | `DEFAULT_CFG`、`RANK`/`LOCAL_RANK`、`LOGGER`、`callbacks` 等 | 日志、并行秩、钩子 |
| 训练辅助 | `check_amp`、`check_train_batch_size`、`ddp_cleanup`、`generate_ddp_command` | AMP 探测、AutoBatch、DDP 命令 |
| Torch 工具 | `ModelEMA`、`EarlyStopping`、`autocast`、`attempt_compile`、QAT 相关、设备选择 | EMA、早停、混合精度、编译与量化 |

---

## 逐段说明

### 类 `BaseTrainer`（第 81–1301 行）

所有任务 Trainer 的抽象基类，定义训练管线骨架；未实现的方法由子类提供。

---

### `BaseTrainer.__init__`（第 135–223 行）

**职责**：合并配置、处理 resume 前置检查、设备与目录、回调、数据集路径解析、清零非 resume 的 `results.csv`。

1. **配置**：`get_cfg(cfg, overrides)` → `self.args`；`check_resume` 可能改写 `args`（真正加载在 `_setup_train`）。
2. **Albumentations**：若 `augmentations` 含 `A.Lambda` 则抛 `TypeError`（无法序列化进 checkpoint）；否则转为 `to_dict` 列表。
3. **设备**：`parse_device` + `select_device`；非 CPU/MPS 时取 `accelerator` 后端。
4. **随机种子**：`init_seeds(seed + 1 + RANK, deterministic=...)`，DDP 各 rank 种子不同。
5. **目录**（rank -1 或 0）：`save_dir`、`weights/`、`args.yaml`；`last.pt`/`best.pt` 路径。
6. **DDP 判定**：`world_size` 由 `device` 字符串中 GPU 数量决定；`ddp = world_size > 1 and LOCAL_RANK == -1`（主进程负责 spawn）。
7. **回调**：拷贝 `_callbacks` 或默认表；非 DDP 主进程注册 integration 并 `on_pretrain_routine_start`（在 `get_dataset` 前保留原始 `data` URI）。
8. **模型路径字符串**：`check_model_file_from_stem`；`torch_distributed_zero_first` 内 `get_dataset()`。
9. **占位**：`ema`、`optimizer`、`scheduler`、`loss` 等指标字段；非 resume 时删除已有 `results.csv`。

---

### `add_callback` / `set_callback` / `run_callbacks`（第 225–236 行）

- `add_callback`：向 `self.callbacks[event]` 追加函数。
- `set_callback`：用单元素列表**替换**该事件全部回调。
- `run_callbacks`：顺序调用，签名均为 `callback(trainer)`。

---

### `train`（第 238–268 行）

**对外训练入口**；**DDP 发生在本方法**，实际 epoch 循环在子进程/单卡调 `_do_train`。

1. `world_size > 1` 且 `batch < world_size` → `ValueError`（每卡至少 1 样本）。
2. **`self.ddp` 为真**：警告 `rect`；`generate_ddp_command` 得到子进程命令；`subprocess.run`；`finally` 里 `ddp_cleanup` 删临时文件。
3. **否则**：直接 `_do_train()`。
4. **`finally`**：`unset_deterministic()`；非 DDP 时 `teardown` 回调。

---

### `_setup_scheduler`（第 270–276 行）

- `cos_lr`：`lf = one_cycle(1, lrf, epochs)`（余弦）。
- 否则：线性从 1 衰减到 `lrf` 的 λ。
- `LambdaLR(optimizer, lr_lambda=lf)` → `self.scheduler`。

---

### `_get_warmup_iterations`（第 278–281 行）

`warmup_epochs`  capped 为 `min(warmup_epochs, epochs-1)`；返回 `round(warmup_epochs * num_batches)` 或 0。用于 batch 循环内 warmup 插值上界。

---

### `_setup_ddp`（第 283–303 行）

**在 `_do_train` 开头、worker 进程内调用**。

1. 解析 `device` 得 `device_type`（npu/xpu/cuda）与 `LOCAL_RANK` 对应 GPU index。
2. `set_device`；CUDA 设 `TORCH_NCCL_BLOCKING_WAIT`；XPU 无 XCCL 则 `RuntimeError`。
3. `init_process_group`：backend hccl/xccl/nccl/gloo，3h timeout；NCCL 且 TORCH_2_8+ 可传 `device_id`。

---

### `_build_train_pipeline`（第 305–343 行）

**在 `_setup_train` 中后期**；建 DataLoader、梯度累积、优化器（若尚无）。

1. 每 rank `batch_size = self.batch_size // world_size`。
2. `train_loader` / `test_loader`（obb/semantic/depth 验证 batch 不 ×2）。
3. BatchNorm 末 batch=1 且过小 imgsz → `ValueError`。
4. `accumulate = max(round(nbs/batch), 1)`；`weight_decay` 按 batch 与 accumulate 缩放；非 `time` 模式 cap accumulate。
5. **优化器**：若 `None`，按数据量与 epochs 估 `iterations` 后 `build_optimizer` + `_setup_scheduler`；否则只更新 weight/muon 组的 `weight_decay`。

---

### `_setup_train`（第 345–473 行）

**训练前一次性装配**；**AMP、DDP 包裹、EMA、断点 resume 均在此完成**，随后才进入 `_do_train` 的 epoch 循环。

#### 段 1：模型与输入布局（347–366）

- `setup_model()` 得 `ckpt`；resume 时重播种子。
- `model.to(device)`；CUDA 可选 `channels_last`（Windows/ROCm 等默认关）。
- `set_model_attributes`；`check_imgsz` 与 `stride`。

#### 段 2：QAT 与编译（368–386）

- `quantize==8`：resume 用 `restore_qat`，否则校准 loader + `prepare_qat`。
- 蒸馏 + compile → 关 compile；`attempt_compile`。

#### 段 3：冻结、AMP、DDP、管道与 EMA（388–473）

- **冻结**：`freeze` 列表 + `.dfl` + 蒸馏 teacher；`requires_grad` 调整；全冻结 → `RuntimeError`。
- **AMP**（418–439）：主进程 `check_amp`；DDP `broadcast` 整型 amp；建 `GradScaler`（npu/xpu/cuda 分支）。
- **蒸馏包装**：非已蒸馏则 `DistillationModel`。
- **DDP**（445–455）：`DistributedDataParallel`，`static_graph` 与 compile 联动，`find_unused_parameters` 与 compile 相反。
- **AutoBatch**：`batch < 1` 且单卡 → `auto_batch()`。
- `_build_train_pipeline`；`get_validator`；`set_class_weights`。
- **EMA**（463）：`ModelEMA(self.model)`。
- rank 0 初始化 `metrics`、可选 `plot_training_labels`。
- `EarlyStopping`；**`resume_training(ckpt)`**（优化器/EMA/scaler/start_epoch）；`scheduler.last_epoch = start_epoch - 1`；`on_pretrain_routine_end`。

---

### `_do_train`（第 475–724 行）

**核心训练循环**：先 DDP init + `_setup_train`，再 `while True` 按 epoch 推进。

#### 段 1：初始化与 epoch 外环（477–503）

- `nb` batches；`nw` warmup iterations；`accumulate` 名义值 `acc0`。
- `on_train_start`；日志数据集规模；`close_mosaic` 时扩展 `plot_idx`。
- `epoch = start_epoch`；`optimizer.zero_grad()`；OOM 重试计数 `_oom_retries = 0`。

#### 段 2：单 epoch 开头（504–524）

- `on_train_epoch_start`；**`scheduler.step()`**（epoch 级 LR，在 optimizer step 之前，警告被忽略）。
- `_model_train()`；DDP **`train_loader.sampler.set_epoch(epoch)`**。
- 近尾声 **关闭 mosaic**：`_close_dataloader_mosaic` + `train_loader.reset()`。
- rank 0：`TQDM` 进度条。

#### 段 3：batch 内循环 — warmup、前向、AMP、反向（525–645）

- `on_train_batch_start`；`ni = i + nb * epoch`（全局 iteration）。
- **Warmup**（`ni <= nw`）：插值 `accumulate`、各 param_group 的 `lr`（bias 从 `warmup_bias_lr`）与 `momentum`。
- **`do_step`**：`ni - last_opt_step >= accumulate`；DDP **`no_sync`**：未 step 且非 compile 时用 `model.no_sync()` 跳过梯度全归约。
- **前向（AMP）**：`autocast(bf16 或 self.amp)` → `preprocess_batch` → compile 时分拆 `forward`+`loss`，否则 `model(batch)`；`loss.sum()`；DDP 下 `loss *= world_size`。
- 首 batch 定 `loss_names`、更新 `metrics` 键、`tloss` 滑动平均。
- **反向**：`scaler.scale(loss).backward()`；OOM/CUDA 错误 → 首 epoch 单卡最多 3 次减半 batch、`_build_train_pipeline`、`break` 重开 epoch。
- **`do_step`** 时：**`optimizer_step()`**（含 unscale、clip、scaler step、**EMA update**）；`time` 模式广播 `stop`。
- rank 0 更新 tqdm；可选 `plot_training_samples`；`on_train_batch_end`；外部 `stop` 可 break。

#### 段 4：epoch 结束 — 验证、NaN、断点、早停（646–711）

- for/else：成功跑完 epoch 清零 OOM 计数；OOM break 则 `continue` 外层 while。
- `criterion.update()`（若存在）。
- 记录 `lr/pg*`；`on_train_epoch_end`；rank 0 **`ema.update_attr`**（yaml/nc 等元数据）。
- DDP 广播 **`stop`**；条件满足则 **`validate()`**（验证前 `_clear_memory`）。
- **`_handle_nan_recovery`**：真则 zero_grad、`continue` 重做 epoch。
- rank 0：`save_metrics` CSV；**`stopper` / `final_epoch` / time** 更新 `stop`；条件满足 **`save_model()`** → `on_model_save`（**断点写入 `last.pt`/`best.pt`/period**）。
- **time 模式**：据平均 epoch 时长重估 `epochs` 并重设 scheduler。
- `on_fit_epoch_end`；内存清理；DDP 再广播 `stop`；`stop` 则 break；`epoch += 1`。

#### 段 5：收尾（713–724）

- 日志总用时；**`final_eval()`**（best.pt 终验）；plots；`on_train_end`；关 loader workers。

**训练要素对照**

| 要素 | 发生位置 |
|------|----------|
| **Epoch** | `_do_train` `while True`，每轮 `scheduler.step()`、验证、save、`epoch += 1` |
| **Batch** | `for i, batch in train_loader`，前向/反向/step |
| **AMP** | `_setup_train` 定 `amp`+scaler；batch 内 `autocast`+`scaler` |
| **EMA** | `_setup_train` 创建；`optimizer_step` 更新；`save_model` 序列化 ema；`validate` 前 DDP 广播 ema buffers |
| **DDP** | `train()` spawn；`_setup_ddp`；`no_sync`/loss 缩放/sampler.set_epoch/多处 `broadcast_object_list` |
| **断点** | `check_resume`/`resume_training`/`_load_checkpoint_state`；epoch 末 `save_model`；NaN 时读 `last.pt` |

---

### `auto_batch`（第 726–743 行）

临时 `cache=False` 建 dataset，估 `max_num_obj` 与 stride 对齐的 `max_imgsz`，调用 `check_train_batch_size` 返回推荐 batch（AutoBatch，仅 `batch<1` 单卡路径使用）。

---

### `_get_memory`（第 745–756 行）

MPS/CUDA 等返回已用显存 GB；`fraction=True` 时返回占用比例（MPS 用 psutil 内存百分比）。

---

### `_clear_memory`（第 758–770 行）

可选阈值：占用未超则跳过；否则 `gc.collect()` 与 `empty_cache`（CPU 直接返回）。

---

### `read_results_csv`（第 772–779 行）

懒导入 polars，读 `self.csv` 为 dict；失败返回 `{}`（供 checkpoint 内 `train_results`）。

---

### `_model_train`（第 781–787 行）

`model.train()`；冻结层路径下的 `BatchNorm2d` 强制 `eval()`（统计量不更新）。

---

### `save_model`（第 789–858 行）

**断点序列化**（rank 0 在 epoch 末由 `_do_train` 触发）。

1. EMA 非有限 → 用 live model 有限权重修补。
2. `qat_state`；EMA deepcopy fp16 contiguous；去 criterion；`nan_to_num_`；`strip_qat`。
3. 单次 `torch.save` 到 `BytesIO`：epoch、best_fitness、stopper、ema、optimizer(fp16)、scaler、train_args、metrics、git 等。
4. 写 `last.pt`；fitness 等于 best 时写 `best.pt`；`save_period` 写 `epoch{N}.pt`。

---

### `get_dataset`（第 860–892 行）

NDJSON 转 YOLO；classify → `check_cls_dataset`；检测族 → `check_det_dataset`；`single_cls` 覆盖 names/nc。

---

### `setup_model`（第 894–934 行）

已 Module 则跳过；`.pt` → `load_checkpoint`；resume 用 ckpt ema/model；`pretrained` 分支；蒸馏 ckpt 重建 `DistillationModel`；否则 `get_model`。

---

### `optimizer_step`（第 936–947 行）

**每个有效 optimize step**：unscale → clip（NPU TORCH_2_0 用 `foreach=False`）→ scaler step/update → zero_grad → **EMA update**。

---

### `preprocess_batch`（第 949–951 行）

基类恒等；子类把 batch 移到 device 等。

---

### `validate`（第 953–971 行）

DDP+EMA 时广播 ema buffers；`self.validator(self)`；`fitness` 从 metrics 或 fallback loss；更新 `best_fitness`。

---

### `get_model` / `get_validator` / `get_dataloader` / `build_dataset`（第 973–987 行）

子类必须实现；基类 `NotImplementedError`。

---

### `label_loss_items`（第 989–993 行）

`loss_items is None` → 带 prefix 的名称列表；否则 округ 字典。

---

### `set_model_attributes`（第 995–997 行）

`model.names = data["names"]`。

---

### `get_class_counts` / `compute_class_weights` / `set_class_weights`（第 999–1028 行）

`cls_pw==0` 跳过；逆频率 ^ cls_pw，无标签类用均值；归一化均值 1；写到 unwrap 后 model（蒸馏取 student）的 `class_weights`。

---

### `progress_string` / `plot_training_samples` / `plot_training_labels`（第 1030–1039 行）

基类空或占位；子类实现表头与绘图。

---

### `save_metrics`（第 1041–1049 行）

追加 `results.csv`：epoch+1、elapsed time、指标列。

---

### `plot_metrics` / `on_plot`（第 1051–1058 行）

`plot_results` 写 results.png；`on_plot` 登记路径与时间戳。

---

### `final_eval`（第 1060–1077 行）

rank 0 `strip_optimizer` last/best；用 **best.pt** 再跑 validator（关 compile）；临时 `epoch+1` 记日志后恢复。

---

### `check_resume`（第 1079–1124 行）

`resume` 真：解析路径/latest run，用 ckpt 的 `train_args` 重建 `args`，仅允许 `allowed` 集合内 overrides 覆盖，其余 warning。

---

### `_load_checkpoint_state`（第 1126–1141 行）

加载 optimizer（同步 fused 标志）、scaler、重建 EMA 并 load_state_dict、updates、best_fitness、stopper 状态。

---

### `_handle_nan_recovery`（第 1143–1181 行）

rank 0 检测 loss/fitness NaN；DDP 广播；首 epoch 或无 last.pt 策略；最多 3 次从 last 恢复 EMA+`_load_checkpoint_state`；`scheduler.last_epoch = epoch-1` 返回 True 触发重做 epoch。

---

### `resume_training`（第 1183–1200 行）

`start_epoch = ckpt.epoch + 1`；断言未训完；`_load_checkpoint_state`；o2o 头 criterion 重置 updates；设 `self.start_epoch`。

---

### `_close_dataloader_mosaic`（第 1202–1208 行）

关 dataset.mosaic 或调用 `close_mosaic(hyp=copy(args))`。

---

### `build_optimizer`（第 1210–1301 行）

#### 段 1：auto 与分组（1228–1257）

`auto`：iterations>1e4 → MuSGD lr0.01，否则 AdamW + 拟合 lr；参数分 weight(decay)、bn、bias、可选 muon 矩阵。

#### 段 2：构造器与 MuSGD 特例（1259–1295）

Adam 族/RMSprop/SGD/MuSGD 参数；MuSGD 对检测头 cv3 等 3×lr 拆组；CUDA+TORCH_2_0 可选 fused Adam(W)。

#### 段 3：日志（1297–1301 行）

打印优化器类型与各 param group 数量。

---

### 类 `MultiTrainer`（第 1304–1468 行）

`data` 为 list/tuple 时 `Model.train()` 自动使用；在 sweep 目录下对每个数据集串行微调，汇总 JSON/柱状图。

---

### `MultiTrainer.__init__`（第 1332–1348 行）

保存 trainer 类、args、基座 model、callbacks；初始化 `trainers`/`metrics`/`mean_metrics`/`save_dir`。

---

### `MultiTrainer.train`（第 1350–1441 行）

1. `get_save_dir(..., name=multitrain)`；CLI 模式（`trainer is None`）将 base model half 存 sweep 下临时 pt。
2. 循环 dataset：派生 `name`/`save_dir`；CLI 子进程 `yolo train` 或 Python `trainer(...).train()`（`get_model` 用基座 yaml+weights）。
3. 收集 metrics（validator 或 ckpt `train_metrics`）；单集失败记 None 不中断。
4. `finally` 删临时 base pt；rank 0 `save_results`、可选 `plot_results`。

---

### `MultiTrainer.save_results`（第 1443–1455 行）

写 `multitrain_results.json`：每 run 指标 + 有效 run 的算术平均 `mean`。

---

### `MultiTrainer.plot_results`（第 1457–1468 行）

按 `TASK2METRIC` 取主指标画跨数据集柱状图。