# `ultralytics/nn/backends/tensorrt.py`

本文件加载 NVIDIA TensorRT 序列化 `.engine`，兼容 TRT 7–9 的 binding API 与 TRT 10+ 的 named tensor API。从文件头读取 JSON 元数据（含 DLA、`nms` 等），在 GPU 上预分配输入输出绑定缓冲；静态形状且非 DLA/嵌入 NMS 时可捕获 CUDA Graph 降低 launch 开销。`forward` 处理动态 batch/尺寸时更新 binding shape 并 `resize_` 输出张量，最终在 CUDA 上返回与 `output_names` 排序一致的 `list[torch.Tensor]`。

## 逐段说明

### `TensorRTBackend` 类（第 18–165 行）

#### `load_model`（第 25–128 行）

- Jetson Py3.8 约束 numpy 版本；`check_tensorrt()`；拒绝 TRT 10.2.0。
- CPU 设备改 `cuda:0`。
- `engine_header` 得偏移与 metadata；`deserialize_cuda_engine`；`create_execution_context`；`apply_metadata`。
- 遍历 IO：记录 `input` 名、动态维、`fp16`；输出名入 `output_names`；每绑定 `torch.empty` on device → `Binding` 入 `OrderedDict`。
- 动态输入：用 profile max shape 初始化 context。
- **CUDA Graph（第 112–126 行）**：TRT10+、静态、非 DLA、非 export nms 时双次 `execute_async_v3` 捕获；失败则 `graph=None`。
- `self.model = engine`（语义为 TRT engine，非 PyTorch module）。

#### `forward`（第 130–165 行）

动态且 `im.shape` 变化：`set_input_shape`/`set_binding_shape`，失败抛 `ValueError`；更新 input binding shape 并 resize 各输出 buffer。断言 batch/spatial 与 binding 一致。无 graph：`execute_v2` 指针列表；有 graph：`copy_` 输入到固定地址后 `replay`。返回 `[bindings[x].data for x in sorted(output_names)]`。

## 与统一接口的对齐

- **加载**：metadata 头与 `.engine` 内嵌字段同步 `end2end`、`imgsz`、`dla`；输出 dtype/shape 来自引擎。
- **前向**：要求输入已在 CUDA，BCHW，与导出时一致；嵌入 NMS 时输出常为 `(B, max_det, 6+)`。
- **输出**：GPU 张量列表，predictor 按 `task` 转 numpy 或直接与 `Results` 对接；动态引擎形状需在 letterbox 允许 profile 范围内。