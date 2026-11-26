# `ultralytics/utils/export/torchscript.py`

将 **PyTorch `nn.Module`**（可含 NMS 包装）经 **JIT trace** 导出为 **`.torchscript`** 归档，可选在 `config.txt` 嵌入 JSON 元数据。

## 逐段说明

### `torch2torchscript`（第 13–39 行）

1. 记录 Torch 版本日志；`output_file` 转 str。
2. `torch.jit._trace.trace(model, im, strict=False, check_trace=False)`。
3. `extra_files` 写入 metadata JSON（若有）。
4. `ts.save` 到目标路径并返回路径字符串。