# `tests/test_cuda.py`

本模块在具备 CUDA 的环境（优先通过 `GPUInfo` 选取空闲 GPU）验证 GPU 训练、导出与推理路径。覆盖 torch 与 `checks` 的 CUDA 一致性、AMP 检查、动态空间 ONNX 后端数值对齐、全任务 ONNX/TensorRT 导出矩阵、GPU 上语义全 ignore 损失的 fp16 有限性、多卡训练与冷进程非零 device、CPU/CUDA 设备切换、导出 TorchScript 跟踪、autobatch、ProfileModels 与 SAM2 GPU 提示。Jetson 缩减任务组合；多项标记 slow。

## 逐段说明

### 模块级 `DEVICES`（第 18–32 行）

有 CUDA 时 Jetson 固定 `[0]`，否则按环境变量空闲比例选最多 2 块空闲 GPU。

### `test_checks`（第 35–38 行）

`torch.cuda` 可用性与设备数与 `tests` 包常量一致。

### `test_amp`（第 41–45 行）

模型迁至 `cuda:DEVICES[0]`，`check_amp` 应通过。

### `test_onnx_dynamic_spatial_outputs`（第 48–64 行）

ResNet18 特征骨干导出动态 H/W ONNX；`ONNXBackend` 在 CUDA 上推理，交换 H/W 输入时 ORT 与包装输出严格一致。

### `test_export_onnx_matrix`（第 67–93 行，`slow`）

全任务×dynamic×batch×simplify×nms 组合（排除 classify+True nms、obb+nms 且 torch<1.13）。导出后 CUDA predict 且 providers 含 CUDA；删除 onnx 文件。

### `test_export_engine_matrix`（第 96–136 行，`slow`）

TensorRT 导出矩阵；TRT<11 跳过 INT8+dynamic。导出后 predict+val；清理 engine 与 ModelOpt 副产物 onnx。

### `test_semantic_loss_all_ignore_amp`（第 139–154 行）

GPU fp16 大 logits + 全 255 mask；loss/items 有限且可 backward。

### `test_train`（第 157–170 行）

灰度与 compile+tile 训练、多卡 device 字符串、断言 `trainer.device.index` 与未改 `CUDA_VISIBLE_DEVICES`；`val=False` 短训仍有 metrics。

### `test_train_cold_process_nonzero_device`（第 173–185 行）

子进程 yolo train `device=max(DEVICES)`，无继承 CUDA_VISIBLE_DEVICES。

### `test_predict_multiple_devices`（第 188–217 行，`slow`）

CPU↔CUDA 多次切换 predict 后 `model.device` 字符串正确。

### `test_track_exported_model`（第 220–227 行）

GPU 导出 torchscript，`track` 带 tile 与 batch=3，断言有 boxes。

### `test_autobatch`（第 230–236 行）

`check_train_batch_size` imgsz=32 结果 >1。

### `test_utils_benchmarks`（第 239–253 行，`slow`）

`ProfileModels` 短跑 isolated 模型。

### `test_predict_sam`（第 256–290 行，`slow`）

SAM2.1 多种 prompt；`SAMPredictor` mobile_sam fp16、`set_image`/`reset_image`。