# `ultralytics/utils/export/rknn.py`

**Rockchip RKNN Toolkit2**：**ONNX→`.rknn`**，可选 INT8 校准、按 head 截断 NPU 子图并导出主机 **`tail.onnx`** 浮点解码。

## 逐段说明

### `_check_rknn_return`（第 11–14 行）

- 返回码非 0/None → `RuntimeError`。

### `onnx2rknn`（第 17–131 行）

**参数校验（53–65）**

- `int8_only` 芯片未 INT8 则 ValueError；INT8 无 dataset 文件则 ValueError。

**环境（67–92）**

- pin `rknn-toolkit2>=2.3.2,<2.5` 与 numpy/protobuf 上限；Colab 屏蔽 `exit`。

**RKNN 配置（94–117）**

- `mean_values=0`、`std_values=255`、`target_platform=name`。
- **INT8 且非 int8_only**：`auto_hybrid_cos_thresh=0.999`。
- **INT8 且提供 head 字符串**：
  - 解析 ONNX：找 head 内末端 Conv（非 dfl） whose 输出被 Reshape/Concat 消费。
  - `outputs` 列表 + 非主输出（如 segment proto）。

**load/build/export（117–126）**

- `config` / `load_onnx(outputs=...)` / `build`（INT8 时 dataset + auto_hybrid）/ `export_rknn`。

**收尾（127–131）**

- **若** 截断输出：`onnx.utils.extract_model` 写 `tail.onnx`。
- metadata.yaml；返回目录。