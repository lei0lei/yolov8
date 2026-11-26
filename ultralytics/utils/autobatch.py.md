# `ultralytics/utils/autobatch.py`

本模块负责在 PyTorch 训练前自动估算 YOLO 模型在给定显存占用比例下的最优 batch size。对外入口是 `check_train_batch_size`，它在与训练一致的 autocast 上下文中深拷贝模型并调用核心函数 `autobatch`。`autobatch` 通过探测若干候选 batch、用 `profile_ops` 测显存占用，再线性拟合推算目标显存对应的 batch。CPU/MPS 或启用 `cudnn.benchmark` 时直接退回默认 batch，避免不可靠的探测。对 RT-DETR 与语义分割会构造特殊前向以测真实 loss 路径。拟合失败或异常时回退默认 batch，并在 `finally` 中清空加速器缓存。

## 逐段说明

### `check_train_batch_size`（第 16～51 行）

- **第 44～51 行**：从模型参数取设备，在 `autocast(enabled=amp, device=...)` 内调用 `autobatch`。
- **第 46 行**：`deepcopy(model).train()` 避免探测改动原模型状态。
- **第 48 行**：当 `0.0 < batch < 1.0` 时把 `batch` 当作显存占用比例，否则使用 `0.6`。
- **第 49～50 行**：将 `max_num_obj`、`dataset_size` 原样传入 `autobatch`。

### `autobatch`（第 54～151 行）— 设备与前缀

- **第 78～80 行**：设置彩色日志前缀并记录 imgsz 与目标显存比例。
- **第 81 行**：`device = next(model.parameters()).device` 确定运行设备。
- **第 82～84 行**：`cpu`/`mps` 时告警并返回 `batch_size`（来自 `DEFAULT_CFG` 或调用方）。
- **第 85～87 行**：CUDA 且 `cudnn.benchmark` 为真时告警并返回默认 batch。

### `autobatch` — 显存信息

- **第 90～92 行**：`get_torch_device_backend` 取加速器；`gb = 1 << 30` 用于 GiB 换算。
- **第 93～94 行**：读设备属性，`t` 为总显存 GiB，`r` 为已保留 GiB。
- **第 97 行**：若有 `mem_get_info`，`f = r + free/GiB` 表示本进程可用显存；否则 `f = t`。
- **第 98 行**：日志输出设备名、总量、保留量、可用量。

### `autobatch` — 候选 batch 与特殊模型包装

- **第 101 行**：总显存小于 16GiB 时候选为 `[1,2,4,8,16]`，否则再加 32、64。
- **第 102～103 行**：`dataset_size > 0` 时过滤掉超过数据集大小的候选。
- **第 104 行**：从 `model.yaml` 取输入通道数，默认 3。
- **第 105～106 行**：`unwrap_model` 得到真实模块；RT-DETR/语义分割需改 profiling 对象。
- **第 107 行**：上述两类模型将 `max_num_obj` 置 0。
- **第 109～113 行**：内联 `ops(x)`：构造 `batch_idx`/`cls`/`bboxes` 与零化图像，调用 `m({img, semantic_mask, ...labels})` 走真实 loss 路径。

### `autobatch` — 探测、拟合与返回

- **第 115～117 行**：为每个候选 batch 建 `torch.empty` 张量列表，调用 `profile_ops(img, ops, n=1, ...)`。
- **第 120～127 行**：从结果筛 `(batch_size, memory)` 点：结果有效、数值型、显存在 `(0,t)`、且显存随 batch 单调（或与首个点对比）。
- **第 128～141 行**：有点则 `np.polyfit` 一阶拟合；`b = int((f * fraction - p[1]) / p[0])` 解最优 batch。
- **第 132～134 行**：找首个失败索引 `oom`，若 `b` 不小于失败点 batch，则退到前一个安全候选。
- **第 135 行**：将 `b` 夹在最小可用探测值与 1024 之间。
- **第 136～137 行**：不超过 `dataset_size`（若给定）。
- **第 139～141 行**：记录预测显存占比并返回 `b`。
- **第 142～144 行**：`except` 任意异常时告警并返回默认 `batch_size`。
- **第 145～146 行**：`finally` 调用 `accelerator.empty_cache()`。
- **第 148～151 行**：无可用拟合点时抛出 `RuntimeError` 并提示释放显存或显式设置 batch。