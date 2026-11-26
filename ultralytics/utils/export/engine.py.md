# `ultralytics/utils/export/engine.py`

TensorRT 导出核心：**PyTorch→ONNX**（`torch2onnx`）、**ONNX→TensorRT `.engine`**（`onnx2engine`），并在 TRT 11 上经 **NVIDIA ModelOpt** 预先写入 FP16/INT8 Q/DQ。含坐标归一化包装类、opset 选择、`modelopt_quantize_onnx` INT8/FP16 路径。

## 逐段说明

### `get_tensorrt_logger`（第 20–25 行）

- `@lru_cache` 单例 `trt.Logger(INFO)`，供 builder 与 parser 共享。

### `_NormalizeCoords`（第 28–59 行）

- **forward**：跑内层 model；对检测输出前 4 通道按输入 w/h 除法归一化。
- **pose 任务**：类分数段与关键点 xy 按 w/h 归一化，visibility 段保持；再 cat。
- 否则 4+nc 后附加剩余通道；tuple 输出保持尾部元组。

### `best_onnx_opset`（第 62–86 行）

- 按 PyTorch 主次版本查表默认 opset（2.x 多钉在 17 以避 ORT CUDA 问题）。
- `min(opset, onnx.defs.onnx_opset_version())` 封顶。

### `torch2onnx`（第 89–128 行）

- 默认输入名 `images`、输出 `output0`；Torch≥2.4 传 `dynamo=False`。
- `torch.onnx.export` 写文件；返回路径。

### `modelopt_quantize_onnx`（第 131–232 行）

- **quantize==8 且无 dataset**：`ValueError`。
- 检查 `nvidia-modelopt[onnx]>=0.44` 且 pin torch 版本。

**INT8 分支（170–211）**

1. 收集至多 512 张校准图，/255 float32。
2. 动态模型传 `calibration_shapes`。
3. 解析 ONNX 图：`transformer` 层索引、`head` 最大层号；SM 12.1 额外排除 `sc/conv`。
4. `modelopt_quantize`：`nodes_to_exclude` 保留 head conv/dfl 与 transformer；`calibration_eps=["cpu"]`。
5. 输出 `.int8.onnx` 路径。

**FP16 分支（213–232）**

1. 用 `bus.jpg` resize/broadcast 到 shape 作 AutoCast 校准（非噪声）。
2. `autocast.convert_to_mixed_precision`：`keep_io_types=True`。
3. 保存 `.fp16.onnx`。

### `onnx2engine`（第 235–496 行）

**初始化（281–324）**

- Jetson/DGX 时 `check_tensorrt` 版本窗；建 `Builder`/`config`/workspace；TRT10+ 无 EXPLICIT_BATCH flag。
- 读 ONNX 判断是否已有 QDQ；`calibrate = int8 且无 QDQ`；无 dataset 则报错。

**DLA（327–341）**

- 非 Jetson、TRT11、或无 fp16/int8 → `ValueError`；设 DLA core 与 `GPU_FALLBACK`。

**TRT11 精度（345–346）**

- fp16/int8 且非已有 QDQ：先 `modelopt_quantize_onnx` 替换 `onnx_file`。

**解析与动态 shape（348–373）**

- `OnnxParser.parse_from_file` 失败 → RuntimeError。
- **dynamic**：optimization profile min/opt/max；老 GPU QDQ 时 min 侧下限 64 或 2×stride。

**Builder 标志（375–382）**

- TRT<11：INT8/FP16 标志；TRT11 依赖图中类型。

**EngineCalibrator（385–480）**

- 仅隐式 INT8 校准：ENTROPY_CALIBRATION_2（DLA）或 MINMAX；`get_batch` 返回 GPU 图像指针。
- 遍历 TRT 层：head 内 Sigmoid→FP32；特定 Conv/ConvTranspose→FP16；设 `OBEY_PRECISION_CONSTRAINTS`/`STRICT_TYPES`。

**序列化与元数据（483–496）**

- `build_serialized_network` 或旧 API；失败 RuntimeError。
- 文件先写 4 字节 metadata 长度 + JSON（若有）再写 engine bytes。