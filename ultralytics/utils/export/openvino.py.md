# `ultralytics/utils/export/openvino.py`

**OpenVINO**：JIT trace 后 `ov.convert_model`，可选 **NNCF INT8**（检测头可 float、RT-DETR 用 TRANSFORMER 预设、NPU 上 detect 可用 PERFORMANCE 对称激活），FP16 由 `compress_to_fp16` 保存。

## 逐段说明

### `torch2openvino`（第 13–88 行）

1. 日志版本；`input_shape` 来自单张或 list 输入。
2. `torch.jit._trace.trace`（`check_trace=False`）→ `convert_model`（动态则用 `example_input`）。
3. **若** `quantize==8`：
   - 探测是否含 `RTDETRDecoder` 决定 `model_type`。
   - **若** `int8_detect`：以最后一个 Sigmoid 为 head 根，`IgnoredScope` 排除 Sigmoid 与 `head.dfl`。
   - `nncf.quantize`：`preset` 在 NPU+detect 非 transformer 时为 PERFORMANCE，否则 MIXED；`subset_size` 用全集长度。
4. **若** `output_dir` 给定：mkdir，`save_model`，`compress_to_fp16=(quantize==16)`。
5. 返回 `ov_model` 对象。