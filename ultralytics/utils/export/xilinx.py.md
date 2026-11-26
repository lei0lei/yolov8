# `ultralytics/utils/export/xilinx.py`

**AMD Xilinx Versal Gen2 NPU**：用 **AMD Quark** 将 FP32 **ONNX** 量化为 **VINT8** ONNX（检测头节点排除保持 float），写 **`vitisai_config.json`** 供 Vitis AI EP 首次加载编译 `.rai`。

## 逐段说明

### `onnx2xilinx`（第 16–92 行）

1. 依赖 `amd-quark`、`onnxruntime` builds；Debian/Ubuntu 可装 `g++`（Quark JIT）。
2. 清空 `output_dir`；从 ONNX 图解析最大 `/model.{i}/` 为 head，收集 head 下节点名至 `exclude`。
3. 临时扩展 PATH（ninja）；`get_default_config("VINT8")`，`enable_npu_cnn`，`nodes_to_exclude`。
4. `ModelQuantizer.quantize_model` 使用 `onnx_calibration_reader`。
5. **finally** 恢复 PATH 并 **删除源 onnx_file**。
6. 写 vitisai_config：`vaip-pass_init` + `vaiml_partition`（device、optimize_level 等）。
7. 返回目录路径。