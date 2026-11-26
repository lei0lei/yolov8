# `ultralytics/utils/export/qnn.py`

**Qualcomm QNN**：将 **ONNX** 经 ORT **QNN QDQ（W8A16）** 量化后，用 **QNN Execution Provider** 在主机上编译嵌入 **context binary** 的 `*_qnn.onnx`，面向指定 Hexagon HTP 架构。

## 逐段说明

### `qnn_library_paths`（第 11–34 行）

- **try** `onnxruntime_qnn.get_library_path/htp_path`。
- **except**：在 onnxruntime capi 目录找 plugin `.so/.dll` 或内置 EP（`ep_lib=None`）；HTP 后端库路径。

### `onnx2qnn`（第 37–167 行）

**路径与日志（85–93）**

- 解析 EP/HTP 库；准备 `pre_file`、`qdq_file`、`ctx_file`。

**输入布局（96–102）**

- **若** ONNX 输入为 NHWC（末维 1/3）：包装 `transform_fn` 在 NCHW 校准后 transpose 到 NHWC。

**量化流水线（104–113，在 try 内）**

1. `quant_pre_process` 写预处理 ONNX。
2. `get_qnn_qdq_config`：激活 QUInt16、权重 QUInt8 + `onnx_calibration_reader`。
3. `quantize` 到 `qdq_file`。

**QNN 编译 session（115–149）**

- `ep_options`：backend_path、finalization mode 3、关 shared memory 与图 I/O 量化卸载（兼容 Android 内置 EP）。
- 从 `QNN_HTP_TARGETS[name]` 写 HTP 选项。
- `SessionOptions`：`ep.context_enable/embed` 指向 `ctx_file`。
- **若** 需注册 plugin：register → 找 QNN devices → `add_provider_for_devices` → `InferenceSession`（**不跑推理**，仅编译）。
- **否则** 传统 providers 列表。
- **finally** unregister plugin。

**清理与元数据（150–167）**

- 删 pre/qdq 中间文件。
- ctx 不存在 → `RuntimeError`。
- 合并缺失 `metadata_props` 到 context 模型并 save。
- 返回 ctx 路径。