# `ultralytics/models/yolo/segment/predict.py`

`SegmentationPredictor` 继承 `DetectionPredictor`：`task=segment`；从模型输出分离 mask 原型，NMS 后按策略生成实例 mask 并写入 `Results.masks`。

## 逐段说明

### `__init__`（34–46）

`self.args.task = "segment"`。

### `postprocess`（48–66）

- `protos`：PyTorch 为 `preds[0][1]`，导出为 `preds[1]`。
- 调用 `super().postprocess(preds[0], ..., protos=protos)` 传原型给 `construct_results`。

### `construct_results`（68–84）

相对检测多 zip `protos`；每图 `construct_result(..., proto)`。

### `construct_result`（86–117）

- 无检测：空 boxes 的 `Results`。
- `boxes` 克隆保留输入像素坐标供 mask 裁剪；`scale_boxes` 写回 `pred[:,:4]`。
- `native = retina_masks or tile` 决定在原图还是输入分辨率上解码 mask。
- 小图密集：`process_mask_native` 或 `process_mask`；否则 `process_mask_local`（ROI 局部解码）。
- `keep`：mask 全零的实例；构造 `Results` 含 masks。
- 未启用特征保存且 `keep` 不全真：`result[keep]` 过滤无 mask 的框。