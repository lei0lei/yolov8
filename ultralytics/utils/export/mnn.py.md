# `ultralytics/utils/export/mnn.py`

通过 MNN **`mnnconvert`** 将 **ONNX** 转为 **`.mnn`**，可选 FP16/INT8 权重量化，元数据写入 `--bizCode` JSON。

## 逐段说明

### `onnx2mnn`（第 12–62 行）

1. 要求 torch≥1.10、ONNX 文件存在；`MNN>=2.9.6`；`import_execstack` 加载带 execstack 的 so。
2. 组装 CLI 风格参数：ONNX→MNN、`bizCode=json.dumps(metadata)`。
3. **若** `quantize==8`：`--weightQuantBits 8`；**若** `16`：`--fp16`。
4. `mnnconvert.convert`；删除 `.__convert_external_data.bin` 临时文件。
5. 返回 `.mnn` 路径。