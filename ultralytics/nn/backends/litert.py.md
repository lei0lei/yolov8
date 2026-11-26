# `ultralytics/nn/backends/litert.py`

本后端运行 **Google LiteRT**（原 TFLite）**`.tflite`** 文件。依赖 **`ai-edge-litert`**（≥2.1.4 或 Intel 路径 ≥2.3.0 + `ai-edge-litert-sdk-intel`）、或 NXP 路径的 **`tflite_runtime`** + Neutron delegate（`libneutron_delegate.so`）。`format` 可为 **`litert`** 或 **`nxp`**。支持 float 图 I/O 与全整型 int8/int16 边界量化/反量化；自动识别 **NCHW**（litert-torch）与 **NHWC**（legacy onnx2tf）。Intel 上 `device` 为 XPU/NPU 时用 **CompiledModel + OpenVINO** 后端；否则 **Interpreter**。输出列表会按任务反归一化 xywh/关键点、调换 segment 输出顺序、NHWC 转 NCHW。

## 逐段说明

### 模块导入（第 1–13 行）

`Path`、`numpy`、`torch`、`LOGGER`、`NUM_THREADS`、`check_requirements`、`BaseBackend`。

### 类 `LiteRTBackend`（第 16–153 行）

#### `__init__`（第 25–35 行）

先设 `self.format = format`（`litert` 或 `nxp`），再 `super().__init__(...)`，以便 `load_model` 中分支 NXP delegate。

#### `load_model`（第 37–88 行）

1. **`tflite_file = Path(weight)`**。
2. **Intel 判定**：从 `self.device` 取类型前缀，`XPU`→`GPU`、`NPU`→`NPU` 映射；`self.device` 强制 `cpu`；`self.intel = (format==litert and device is not None)`。
3. **分支 `self.intel` 真**：
   - 检查双包版本；`CompiledModel.from_file`，`HardwareAccelerator.NPU|CPU`，`IntelOpenVinoOptions(graph_backend=backend)`。
   - 取首个 signature；创建 input/output buffers；`input_details`/`output_details` 来自 tensor details，并把 `dtype` 转为 `np.dtype`。
4. **分支 `else`**：
   - **`format == "nxp"`**：`tflite_runtime.Interpreter` + `load_delegate("libneutron_delegate.so")`。
   - **否则**：`ai_edge_litert.interpreter.Interpreter`，`num_threads=NUM_THREADS`，`allocate_tensors()`，读 get_input/output_details。
5. **布局**：`read_metadata`；`nhwc = (input shape 末维 == metadata channels)`。
6. **`apply_metadata(metadata)`**。

#### `forward`（第 90–153 行）

**输入预处理**：

1. `im.cpu().numpy()` BCHW。
2. **`nhwc`**：`transpose(0,2,3,1)`；`h,w` 取自 shape `[1:3]`，否则取自 `[2:4]`。
3. **整型输入**：若 `input_details[0].dtype` 为 int8/int16，用 `quantization` 的 scale/zero_point 量化并 clip 到 dtype 范围。

**执行**：

- **`intel`**：`input_buffers[0].write(contiguous im)` → `run_by_index(0, ...)`。
- **否则**：`set_tensor` + `invoke()`。

**逐输出 `enumerate(output_details)`**：

1. **读张量**：intel 从 buffer `read`+reshape，否则 `get_tensor`。
2. **`task==semantic` 且 `x.ndim==3`**：legacy 已 argmax 的 `[B,H,W]`，直接 `append`，跳过反归一化。
3. **整型输出**：float32 反量化 `(x - zp) * scale`。
4. **反归一化坐标**：
   - `ndim==3` 且 `not end2end` 且 `head != RTDETRDecoder`：`denormalize(x, w, h)`。
   - `ndim==3` 且 `end2end` 且 `nhwc`：`denormalize(..., end2end=True)`。
5. `y.append(x)`。

**后处理输出列表**：

- **segment**：若 `y[0].ndim==4`（检测与 proto 顺序反了），`y = [y[1], y[0]]`。
- **`nhwc`**：segment 的 proto `y[1]` 或 semantic/depth 单输出 4D 时 `transpose(0,3,1,2)` → NCHW。

返回 `list[np.ndarray]`。