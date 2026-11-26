# `examples/python/ONNXRuntime/main.py`

本脚本是「纯 ONNX Runtime、无 PyTorch」推理示例：从 Ultralytics 导出的 ONNX 读取 custom metadata（任务、类名、输入尺寸、可选关键点维度），自动选择 CUDA 或 CPU 执行提供器，经 `yolo_common` 完成前处理、会话 `run` 与后处理，最后可视化或保存。RT-DETR 通过 metadata 中 `head` 或 `task` 识别，预处理策略与 letterbox 检测不同。入口在模块顶层顺序执行，无 `main()` 函数。

## 逐段说明

### 初始化与会话（第 3–27 行）

将 `examples/python/common` 加入 `sys.path`，`parse_args("yolo26n.onnx")` 解析默认模型与源。按可用性构造 ORT `providers` 列表并创建 `InferenceSession`。从 `custom_metadata_map` 用 `ast.literal_eval` 解析 `names`、`imgsz`、可选 `kpt_shape`；输入张量类型决定是否 FP16 blob。`task` 来自 CLI 覆盖或 metadata（RT-DETR 特殊分支）。打印模型路径、任务与类数。

### 推理与输出（第 29–32 行）

`cv2.imread` 读图；`preprocess` 返回 blob 与 pad；`session.run` 得原始输出；`postprocess` 解码框/掩膜/姿态等；`finish` 按 CLI 显示或写盘。