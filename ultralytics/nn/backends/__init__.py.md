# `ultralytics/nn/backends/__init__.py`

本文件是 YOLO 推理后端的包入口，统一导出各部署格式对应的 `*Backend` 类。上层 `AutoBackend` 根据权重后缀或路径选择具体实现，而使用者也可直接 `from ultralytics.nn.backends import ONNXBackend` 等方式单独实例化。所有后端均继承 `BaseBackend`，共享元数据读取、设备与 FP16 等公共语义。导出列表 `__all__` 与 import 一一对应，便于类型检查与公共 API 文档生成。本模块不包含运行时逻辑，仅承担注册与再导出职责。

## 逐段说明

### 模块文档字符串（第 1–7 行）

说明包用途：为多种深度学习框架与硬件加速器提供模块化推理后端，每个后端实现 `BaseBackend`，可独立使用或通过 `AutoBackend` 自动路由。

### 后端类导入（第 9–28 行）

按功能域导入具体后端：`AscendBackend`、`AxeleraBackend`、`BaseBackend`（抽象基类）、`CoreAIBackend`、`CoreMLBackend`、`DeepXBackend`、`ExecuTorchBackend`、`HailoBackend`、`LiteRTBackend`、`MNNBackend`、`NCNNBackend`；ONNX 相关为 `ONNXBackend`、`ONNXIMXBackend`、`STEdgeAIBackend`；`OpenVINOBackend`、`PaddleBackend`；PyTorch 系为 `PyTorchBackend`、`TorchScriptBackend`；`QNNBackend`、`RKNNBackend`、`TensorFlowBackend`、`TensorRTBackend`、`TritonBackend`。

### `__all__`（第 30–54 行）

公开 API 元组，名称与上述导入一致（不含模块级辅助函数）。保证 `from ultralytics.nn.backends import *` 与显式导入行为一致。