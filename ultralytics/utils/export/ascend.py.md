# `ultralytics/utils/export/ascend.py`

调用华为 **CANN ATC**，将 **ONNX** 编译为 Ascend 离线 **`.om`**（目录内），固定 NCHW、`force_fp16` 输入。

## 逐段说明

### `_check_atc`（第 14–21 行）

- `shutil.which("atc")` 失败则 `FileNotFoundError` 与文档链接。

### `onnx2ascend`（第 24–72 行）

1. `_check_atc`；清空并重建 `output_dir`。
2. 组装 `atc` 参数：framework=5（ONNX）、input_shape、soc_version、FP16 精度模式。
3. 在临时 cwd 运行（避免 kernel_meta 污染用户目录）。
4. 可选 `metadata.yaml`；返回目录路径。