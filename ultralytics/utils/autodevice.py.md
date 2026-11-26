# `ultralytics/utils/autodevice.py`

本模块通过 NVIDIA NVML（`pynvml`）封装 GPU 状态查询与「空闲 GPU」选择，供多进程或自动选卡场景使用。核心类 `GPUInfo` 在初始化时尝试 `nvmlInit`，失败则记录警告并保持 `nvml_available=False`，避免缺库时崩溃。`refresh_stats` 遍历设备填充利用率、显存、温度、功耗等字典列表。`select_idle_gpu` 按空闲显存比例与空闲利用率阈值筛选，再按利用率升序、空闲显存降序排序，并用随机 tie-break 减轻多进程同时抢同一批卡的竞态。模块末尾 `__main__` 块提供命令行式演示。

## 逐段说明

### 类 `GPUInfo`（第 11～198 行）

#### `__init__`（第 44～58 行）

- **第 46～48 行**：初始化 `pynvml=None`、`nvml_available=False`、`gpu_stats=[]`。
- **第 50～56 行**：`try` 导入并 `nvmlInit()`，成功则置 `nvml_available=True` 并 `refresh_stats()`。
- **第 57～58 行**：任意异常仅 `LOGGER.warning`，不抛出。

#### `__del__`（第 60～62 行）

- 析构时调用 `shutdown()`，确保 NVML 关闭。

#### `shutdown`（第 64～71 行）

- **第 66～70 行**：若 NVML 可用则 `nvmlShutdown()`，异常静默；最后 `nvml_available=False`。

#### `refresh_stats`（第 73～84 行）

- **第 75 行**：清空 `gpu_stats`。
- **第 76～77 行**：NVML 不可用则直接返回。
- **第 80～81 行**：`nvmlDeviceGetCount()` 后对每个索引调用 `_get_device_stats` 并 extend。
- **第 82～84 行**：查询异常时警告并清空列表。

#### `_get_device_stats`（第 86～111 行）

- **第 88～90 行**：取 handle、显存信息、利用率。
- **第 92～97 行**：内部 `safe_get` 在 NVML 调用失败时返回 `default`（默认 -1），可选 `divisor` 做单位换算。
- **第 99 行**：温度类型常量 `NVML_TEMPERATURE_GPU`。
- **第 101～111 行**：组装字典：索引、名称、GPU 利用率、显存 used/total/free（MiB）、温度、功耗 draw/limit（mW→W）。

#### `print_status`（第 113～133 行）

- **第 115 行**：先 `refresh_stats()`。
- **第 116～118 行**：无数据则警告返回。
- **第 121～123 行**：按最长 GPU 名动态列宽，打印表头与分隔线。
- **第 125～131 行**：逐卡格式化 Util/Mem/Temp/Pwr，无效指标显示 N/A。
- **第 133 行**：打印底部分隔线。

#### `select_idle_gpu`（第 135～198 行）

- **第 159～160 行**：断言 `min_memory_fraction`、`min_util_fraction` 均 ≤ 1.0。
- **第 161～164 行**：记录筛选条件并打日志。
- **第 166～167 行**：`count <= 0` 返回空列表。
- **第 169 行**：`refresh_stats()`。
- **第 170～174 行**：无 NVML 统计时 import torch，警告后按 `indices` 或 `range(device_count())` 取前 `count` 个（不做空闲校验，如 AMD）。
- **第 177～183 行**：筛选：`indices` 限制、`memory_free/total >= min_memory_fraction`、`(100 - utilization) >= min_util_fraction*100`。
- **第 186 行**：排序键为 `(utilization, -memory_free, random.random())`，利用率低优先，同利用率则空闲显存多优先。
- **第 189 行**：取前 `count` 个索引。
- **第 191～196 行**：有选中则记录数量不足警告（若适用）并 info 选中设备；否则警告无 GPU 达标。

### 模块 `__main__`（第 201～216 行）

- **第 202～204 行**：示例阈值与要选 GPU 数量。
- **第 206～216 行**：创建 `GPUInfo`、打印状态、调用 `select_idle_gpu` 并打印 `cuda:{idx}` 设备串。