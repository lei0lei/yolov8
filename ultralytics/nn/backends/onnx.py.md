# `ultralytics/nn/backends/onnx.py`

本文件以 ONNX Runtime（及 OpenCV DNN）运行 `.onnx` 导出图，并包含 Sony IMX、ST Edge AI 等专用子类。模块级函数负责 AMD MIGraphX EP 注册、编译缓存目录与 session 创建。`ONNXBackend` 按设备选择 CUDA、CoreML EP、Vitis AI、Windows ML 或 CPU，静态形状下对 CUDA/ROCm 启用 IO binding 以减少 H2D/D2H。`STEdgeAIBackend` 在 host 上反归一化框/关键点并将 depth 从 log 转米制。`ONNXIMXBackend` 在 CPU 上加载 MCT 量化模型并把多输出拼成与 Ultralytics 检测布局一致的张量。

## 逐段说明

### 模块常量（第 27–37 行）

`ROCM_EXTRA_INDEX`、`ROCM_EP_PACKAGES`：MIGraphX 插件与库版本钉扎。`MIGRAPHX_CACHE_MODELS=8`：LRU 缓存模型数。`WINML_SUPPORTED`：Win11 24H2+ 且 Python≥3.10。

### `_register_migraphx_ep`（第 40–67 行）

预加载 `libonnxruntime`/`libmigraphx_c`（RTLD_GLOBAL），`register_execution_provider_library`，返回 EP 名称。

### `_migraphx_cache_root`（第 70–79 行）

`lru_cache` 解析 `ORT_MIGRAPHX_CACHE_DIR` 或 `USER_CONFIG_DIR/migraphx_cache`。

### `_migraphx_cache_dir`（第 82–100 行）

按模型 SHA256 前 16 位建子目录，`utime` 标记 MRU，超出 8 个则 `rmtree` 最旧目录。

### `_load_migraphx_session`（第 103–142 行）

`ThreadingLocked`；设置 `MIGRAPHX_DISABLE_WINOGRAD`；写 `ORT_MIGRAPHX_CACHE_DIR`；`enable_fallback=0` 以便失败时由上层回退 CPU。返回 MIGraphX EP 的 `InferenceSession`。

### `_ORT_DTYPES`（第 145–154 行）

ORT 输出类型字符串到 `(torch.dtype, np.dtype)` 映射，供 IO binding 预分配。

### `ONNXBackend` 类（第 157–361 行）

#### `__init__`（第 165–187 行）

`format` 必须为 `onnx` 或 `dnn`；保存 `session_options`、`cuda_options` 后 `super().__init__`。

#### `load_model`（第 189–317 行）

**设备与非目录权重（第 195–202 行）**：非 `torch.device` 字符串改 CPU；`read_metadata`；目录导出取首个 `*.onnx`。

**OpenCV DNN 分支（第 204–209 行）**：`cv2.dnn.readNetFromONNX`，`self.net`。

**ONNX Runtime 分支（第 210–317 行）**：

- ROCm 时尝试 `_load_migraphx_session`，失败则 CPU。
- 否则建 `SessionOptions`（可设 `intra_op_num_threads`）；Linux CUDA 可 `preload_dlls`。
- 按可用 EP 选 CUDA、MPS+CoreML、VitisAI（需 `vitisai_config.json`）、WinML 或 CPU。
- `InvalidProtobuf` 转为友好 `TypeError`。
- 记录 `output_names`、`dynamic`（batch 维为字符串）、`fp16`（输入类型）。
- 静态全整型输出 shape 时：`use_io_binding=True`，预分配 GPU 张量并 `bind_output`（ROCm 用 DLPack）。

#### `forward`（第 319–361 行）

- **dnn**：`setInput` numpy，`forward` 未连接层名，返回 tuple。
- **多输入 dict**：键名对 session 输入，`session.run`。
- **IO binding**：`contiguous` 输入，`bind_input`/DLPack，`run_with_iobinding`，返回 `self.bindings`（GPU 张量列表）。
- **否则**：`session.run`，numpy 列表。

### `STEdgeAIBackend`（第 364–386 行）

#### `forward`（第 372–386 行）

`super().forward` 后：`task=="depth"` 对 `y[0]` 做 `exp`；`detect/segment/pose/obb` 调用 `denormalize(y[0], w, h)`。对齐 ST Edge 导出时 [0,1] 归一化坐标。

### `ONNXIMXBackend`（第 389–443 行）

#### `load_model`（第 396–419 行）

安装 MCT/edge-mdt/onnxruntime-extensions；目录下 `*.onnx`；`mctq.get_ort_session_options()`；CPU EP；`apply_metadata`。

#### `forward`（第 421–443 行）

按 `task` 拼接输出：**detect** → `[boxes, conf, cls]` 沿最后一维 concat；**pose** → 加 `kpts`；**segment** → det 向量 + proto `y[4]`；其余原样返回。与 predictor 期望的 `(B,N,C)` 布局对齐。

## 与统一接口的对齐

- **加载**：`apply_metadata` 设置 `task`、`imgsz`、`end2end`；动态 batch 时 `dynamic=True`。
- **前向**：标准单输入 BCHW；返回 numpy 或 GPU 绑定张量列表，顺序与 `output_names` 一致。
- **输出**：无内置 NMS；嵌入 NMS 的 ONNX 输出 `(B,N,6+)` 由上层按 `end2end` 处理；IMX/STEdge 子类在 host 上做格式修正。