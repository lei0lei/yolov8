# `ultralytics/models/yolo/world/val.py`

`WorldValidator` 继承 `DetectionValidator`，不改 mAP 流程；在**独立** `val()` 时把数据集类名的文本嵌入注入 `WorldModel`，避免默认 COCO 80 类与 LVIS 等数据集不匹配导致指标为零。

## 逐段说明

### `__call__`（20–40）

- `trainer is None`（非训练内验证）：
  - `select_device` 设 `self.device`。
  - 若 `model` 非 `nn.Module`，`load_checkpoint` 并 `eval().to(device)`。
  - `convert_ndjson_to_yolo_if_needed` 规范化 data。
  - 从 `check_det_dataset` 取 names，截断 `/` 前主类名。
  - 与 `model.names` 列表比较；若不同则保存 `(names, txt_feats, nc)`，调用 `set_classes(names)` 并 `names=dict(enumerate(names))`，`try/finally` 里 `super().__call__` 后恢复原状态，避免泄漏给调用方。
- 训练内验证：直接 `super().__call__(trainer, model)`（类由 `on_pretrain_routine_end` 已设）。