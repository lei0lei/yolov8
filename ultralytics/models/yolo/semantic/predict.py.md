# `ultralytics/models/yolo/semantic/predict.py`

`SemanticSegmentationPredictor` 继承 `BasePredictor`（非检测）：输出单通道类 ID 图 `Results.semantic_mask`，无上采样到框。

## 逐段说明

### `__init__`（25–34）

`task=semantic`。

### `_class_map_dtype`（36–39，静态）

类数 ≤256 用 `uint8`，≤32768 用 `int16`，否则 `int32`。

### `postprocess`（41–89）

- tuple/list 取 `[0]`。
- 非 list `orig_imgs` 转 numpy BGR。
- `classes` 过滤：多类且 `args.classes` 设定时在设备上建索引张量。
- 逐图：`class_map_input = pred.ndim==2`（导出已 ArgMax）。
- 类图分支：`unsqueeze` 为 4D；与 `img` 尺寸不一致则 nearest 插值；`scale_masks` 到原图；按最大值选 dtype。
- logits 分支：双线性对齐输入图；`scale_masks` 去 letterbox 到原图；`nc>1` 时 argmax/max(dim0)，单通道二值 `gt(0)`。
- `classes` 过滤：不在集合内像素标 255（ignore）。
- `Results(..., semantic_mask=class_map)`。