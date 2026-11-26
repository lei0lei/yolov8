# `ultralytics/nn/backends/openvino.py`

本文件通过 Intel OpenVINO 运行 `*_openvino_model` 目录中的 IR（`.xml` + `.bin`）。加载阶段将 PyTorch 设备名映射为 OpenVINO 的 CPU/GPU.N/NPU/AUTO，并针对 Windows、ARM64、NPU、AMX+INT8 动态图等场景做编译配置修补。推理在 CPU 上使用同步 `compile_model` 调用，便于 predictor 预取下一批；动态 INT8 时在 `forward` 内按实际输入 reshape 并重新 compile。输出为 `list[np.ndarray]`，与 ORT CPU 路径类似，由 `AutoBackend` 转 torch 并解码。

## 逐段说明

### `OpenVINOBackend` 类（第 18–123 行）

#### `load_model`（第 25–104 行）

- 要求 `openvino>=2024.0.0`；`ov.Core()`。
- Windows CPU：强制 FP32 推理精度；`ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16` 规避 AMX 虚拟机 fault。
- 设备：`CPU`/`XPU`→`GPU.*`/`NPU` 或 `AUTO`/`CPU` 回退；`self.device` 固定为 `torch.device("cpu")`（OV 自管设备）。
- 权重：非文件则 `glob *.xml`；`read_model` + 同 stem `.bin`；`apply_metadata`。
- **AMX INT8 动态图（第 59–74 行）**：CPU/AUTO + FakeQuantize + 动态输入 + Linux `amx_int8` 时保存 `read_model` partial，关闭 `dynamic` 标志，改在 forward 按形状重编译。
- **编译配置**：`PERFORMANCE_HINT=LATENCY`；ARM64 CPU 加 ACCURACY+f32；NPU 分支：batch 钉 1、动态导出 pin `imgsz`、depth 任务剪去错误上采样、classify 可开 `NPU_TURBO`。
- `compile_model` partial、`ov_compiled_model`；`prefetch=(device_name=="CPU")`。

#### `forward`（第 106–123 行）

输入转 `float32` numpy。若 `read_model` 且当前 partial shape ≠ `im.shape`：reshape 后重新 `compile_model`。返回 `list(ov_compiled_model(im).values())`，多输出顺序与 IR 输出端口一致。

## 与统一接口的对齐

| 阶段 | 行为 |
|------|------|
| 加载 | 侧车 `metadata.yaml` 提供 `names`、`task`、`stride`；NPU 可能改写 `batch`/`dynamic`。 |
| 前向 | BCHW numpy；不在此做 NHWC 转换（导出 IR 通常为 NCHW）。 |
| 输出 | 多输出 list，检测多为 `(B,4+nc,N)` 或分割第二路 proto；与 PyTorch 导出节点对齐。 |