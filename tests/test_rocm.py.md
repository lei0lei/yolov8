# `tests/test_rocm.py`

本文件验证 AMD ROCm/HIP 环境下 YOLO 导出与 ONNX Runtime 推理路径。首个矩阵测试对全部官方任务导出小尺寸 ONNX，并在 `device=0` 上 predict，断言 MIGraphX 执行提供器出现在 ORT session 中。第二个测试在 ROCm 机器上显式 `device="cpu"` 推理，确认回退到纯 CPUExecutionProvider。两项均在无 ROCm 时整模块跳过；矩阵项额外标记为 slow。

## 逐段说明

### `test_rocm_migraphx_matrix`（第 15–23 行）

参数化 `task` 为 `TASKS`。使用 `isolated_model_path` 加载 `TASK2MODEL[task]`，导出 `onnx`（`imgsz=32`，`device=0`），再对 `SOURCE` 推理。断言返回 truthy，且 `get_providers()` 含 `"MIGraphXExecutionProvider"`。

### `test_rocm_cpu_fallback`（第 26–32 行）

用 `isolated_model` fixture 导出 ONNX，加载后对 `SOURCE` 设 `device="cpu"`。断言推理成功且 providers 列表仅含 `CPUExecutionProvider`。