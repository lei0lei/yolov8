# `ultralytics/utils/export/__init__.py`

导出子包聚合入口：从各格式模块 re-export **PyTorch→目标** 与 **ONNX→目标** 的转换函数，供 `ultralytics.engine.exporter.Exporter` 按 `format=` 分派调用。

## 逐段说明

### 相对导入（第 3–20 行）

按后端拉取符号，对应关系概览：

| 符号 | 典型转换链 |
|------|------------|
| `torch2onnx`, `onnx2engine` | PyTorch → ONNX → TensorRT `.engine` |
| `torch2torchscript` | PyTorch → `.torchscript` |
| `torch2openvino` | PyTorch → OpenVINO IR |
| `torch2coreml`, `torch2coreai` | PyTorch → CoreML / Apple Core AI |
| `torch2paddle`, `torch2ncnn` | PyTorch → Paddle / NCNN |
| `onnx2saved_model`, `keras2pb`, `tflite2edgetpu` | ONNX → TF SavedModel / `.pb` / Edge TPU |
| `torch2litert`, `torch2executorch`, `torch2imx` | 移动端与 NPU 运行时 |
| `onnx2mnn`, `onnx2rknn`, `onnx2qnn`, `onnx2ascend`, `onnx2xilinx`, `onnx2deepx` | ONNX 后处理到各厂商格式 |
| `torch2axelera` | PyTorch → Axelera `.axm` |

### `__all__`（第 22–44 行）

- 与上述导入列表一致的公开 API 名称元组。