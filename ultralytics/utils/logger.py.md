# `ultralytics/utils/logger.py`

本模块提供训练侧两类日志设施：`ConsoleLogger` 捕获 stdout/stderr 与 Ultralytics logging，去重、分批刷新到文件或 HTTP API，并与 `TQDM.progress` 协同将进度条帧当状态而非重复日志行；`SystemLogger` 周期性采集 CPU/RAM/磁盘/网络及 NVIDIA 或 AMD GPU 指标，供 Platform 等监控。`_DriveInfo` 封装多平台「用户可见挂载点」发现逻辑。

## 逐段说明

### 类 `ConsoleLogger`（第 17～273 行）

#### `__init__`（第 48～85 行）

- HTTP 明文 warning；`destination` 区分 API（http/https）或 `Path`。
- `batch_size=max(1,...)`、`flush_interval`、`on_flush` 回调。
- 保存原 stdout/stderr；缓冲与锁；`chunk_id` 毫秒时间戳；进度条状态 dict `progress`/`progress_sent`。

#### `start_capture`（第 87～110 行）

- 已 active 或 `RANK` 非 -1/0 则 return。
- 替换 stdout/stderr 为 `_ConsoleCapture`；添加 ultralytics `_LogHandler`。
- 启动守护线程 `_flush_worker`（若未存活）。

#### `stop_capture`（第 112～132 行）

- 清空 capture callback；恢复标准流；移除 handler；清 progress 后 `_flush_buffer()`。

#### `_queue_log`（第 134～182 行）

- 非 active return；处理 `\r` 与 `\x1b[K`。
- **第 146～151 行**：`bar_id` 非空：非空 frame 写入 `progress[id]`；空 frame 弹出 closed 条成为日志行。
- 分行 rstrip；0.1s 内完全相同行去重。
- 无 `[20` 前缀则加本地时间戳。
- 达 `batch_size` 在锁外 `_flush_buffer` 防死锁。

#### `_flush_worker`（第 184～189 行）

- active 时每隔 `flush_interval` sleep 后 flush。

#### `_flush_buffer`（第 191～214 行）

- 无新行且 progress 未变则 return；复制 lines、清 buffer、同步 `progress_sent`、递增 `chunk_id`。
- 可选 `on_flush(content, line_count, chunk_id)`；有 destination 且 lines 非空则 `_write_destination`。

#### `_write_destination`（第 216～229 行）

- API：`requests.post` JSON timestamp+message；文件：mkdir 后 append UTF-8。

#### 嵌套类 `_ConsoleCapture`（第 231～259 行）

- `write` 写原流并 `callback(text, None)`。
- `progress` 写 frame 并 `callback(frame, bar_id)`。
- `flush`/`isatty` 委托原流。

#### 嵌套类 `_LogHandler`（第 261～273 行）

- `emit` 格式化 record 加换行送 callback。

### 类 `_DriveInfo`（第 276～408 行）

#### `mounts`（第 289～313 行）

- `psutil.disk_partitions`；默认仅当前 cwd 所在挂载。
- `all_drives` 时过滤 dontbrowse；≤1 个挂载则当前挂载；否则 macOS/linux/windows 专用 getter 或回退排序列表。

#### `_sort`（第 315～319 行）

- 排除 `/boot/`、`/efi/` 前缀；`/` 优先。

#### `_current_mount`（第 321～336 行）

- cwd resolve 失败则 Windows `C:\` 或 `/`。
- 最长匹配 mountpoint 作为当前盘。

#### `_macos_mounts`（第 338～367 行）

- `diskutil list` 物理盘集合；每分区 `diskutil info` plist，过滤 APFS 卷组重复，仅物理盘相关挂载。

#### `_linux_mounts`（第 369～393 行）

- `lsblk --json` 递归 visit 磁盘树，每物理盘取最短 mountpoint。

#### `_windows_mounts`（第 395～408 行）

- PowerShell `Win32_LogicalDisk` DriveType=3 固定盘列表。

### 类 `SystemLogger`（第 411～700 行）

#### `__init__`（第 454～487 行）

- import psutil；ROCm 时跳过 NVML 初始化 AMD `_init_amd`。
- NVML 成功缓存 driver/cuda 版本字符串。
- `net_start`/`disk_start`、`_DriveInfo.mounts`、`gpus` 过滤列表；速率计算用 `_prev_*`。

#### `_init_nvidia`（第 489～505 行）

- macOS 直接 False；pynvml init 失败且 CUDA 可用则 warning。

#### `_init_amd`（第 507～530 行）

- `amdsmi_init`；按 PCI BDF 与 `torch.cuda.get_device_properties` 匹配 handle 字典；异常 warning 空 dict。

#### `get_metrics`（第 532～651 行）

- 读 net/disk_io/memory；`elapsed` 防除零。
- **rates=True**：disk/network 为 MB/s 增量率；否则为自初始化起累计 MB。
- 遍历 mounts（失败回退根盘）得 used/total GB。
- 合并 nvidia_versions、cpu%、ram%、disk 列表、disk_io、空 gpus。
- 更新 `_prev_net/disk/time`。
- 有 NVML 则 `_get_nvidia_metrics` 否则 AMD dict 填入 gpus。

#### `_get_nvidia_metrics`（第 653～675 行）

- 遍历 `self.gpus` 或全部设备：util、显存百分比、温度、功耗 W。

#### `_get_amd_metrics`（第 677～700 行）

- 每 handle：vram 百分比、功耗、edge/hotspot 温度、gfx_activity 为 usage。

### 模块 `__main__`（第 703～743 行）

- `SystemLogger(all_drives=True)` 循环 `get_metrics` 清屏打印直至 Ctrl+C。