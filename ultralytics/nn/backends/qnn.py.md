# `ultralytics/nn/backends/qnn.py`

本后端在 **Qualcomm Snapdragon**（Android、WoS、Qualcomm Linux）上运行 Ultralytics QNN 导出的 **`*_qnn.onnx`**（QNN 上下文二进制经 ORT 封装）。依赖 **`onnxruntime-qnn`** 与 **`onnxruntime`**；通过 `ultralytics.utils.export.qnn.qnn_library_paths()` 解析 QNN EP 插件与 **HTP** `backend_path`。加载时注册 **QNNExecutionProvider** 并创建 `InferenceSession`；根据输入 shape 判断 **NHWC**。输入 **BCHW [0,1]**，NHWC 时 `permute(0,2,3,1)`；`session.run` 按输出名列表返回。

## 逐段说明

### 模块导入（第 1–12 行）

`Path`、`torch`、`LOGGER`、`check_requirements`、`BaseBackend`。

### 类 `QNNBackend`（第 15–76 行）

#### `load_model`（第 23–63 行）

1. `check_requirements("onnxruntime-qnn")`；`import onnxruntime`。
2. `onnx_file = Path(weight)`；日志。
3. **`ep_name = "QNNExecutionProvider"`**；`ep_library, htp_backend = qnn_library_paths()`；`ep_options = {backend_path: htp_backend}`；`SessionOptions()`。
4. **分支 `ep_library` 非空**（需动态注册插件）：
   - `register_execution_provider_library(ep_name, ep_library)`。
   - `get_ep_devices()` 筛 `ep_name`；**空列表** → `OSError`（非 Snapdragon 或未装 qnn 包）。
   - `add_provider_for_devices(devices, ep_options)` → `InferenceSession(onnx_file, sess_options=options)`。
5. **分支 `ep_library` 为空**（QNN 已内置 ORT）：
   - `InferenceSession(..., providers=[ep_name], provider_options=[ep_options])`。
6. `output_names` 来自 `get_outputs()` 名称。
7. **NHWC 检测**：首输入 shape 长度为 4 且 `shape[3] in {1,3}` 且 `shape[1] not in {1,3}` → `self.nhwc=True`。
8. `apply_metadata(read_metadata(onnx_file))`。

#### `forward`（第 65–76 行）

1. **`self.nhwc` 真**：`im = im.permute(0,2,3,1)`。
2. `return session.run(output_names, {input_name: im.cpu().numpy()})`——返回 ORT 输出列表，顺序与 `output_names` 一致；无量化、无主机解码。