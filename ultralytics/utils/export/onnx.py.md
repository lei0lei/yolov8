# `ultralytics/utils/export/onnx.py`

ONNX Runtime **静态 INT8 量化**辅助：提供校准数据读取器与 `quantize_static` 封装，将 **FP32 ONNX** 转为带 Q/DQ 的 **INT8 ONNX**（默认仅量化 Conv/Gemm/MatMul，保留检测头浮点解码）。

## 逐段说明

### `onnx_calibration_reader`（第 6–34 行）

- 返回 `CalibrationDataReader` 子类实例。
- `__init__`：`iterator = iter(dataset)`。
- `get_next`：batch 耗尽返回 None，否则 `{input_name: transform_fn(b)}`。
- `rewind`：重置迭代器供多轮校准。

### `onnx_int8_quantize`（第 36–80 行）

1. `onnx.load`：若非 `whole_graph`，`exclude` 为图中非 Conv/Gemm/MatMul 的节点名（避免检测头单 scale 压扁分数；同时避免 attention Softmax 未校准崩溃）。
2. 据 opset≥13 决定 `per_channel`。
3. `quantize_static`：校准 reader、`nodes_to_exclude`、`per_channel`。
4. 返回输出路径字符串。