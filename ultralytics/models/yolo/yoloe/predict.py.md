# `ultralytics/models/yolo/yoloe/predict.py`

视觉提示（VP）推理：`YOLOEVPDetectPredictor` 在检测前把 bbox/mask 光栅化为 `visuals` 张量，经 `inference(..., vpe=visuals)` 送入模型；`YOLOEVPSegPredictor` 多重继承分割后处理。

## 逐段说明

### `YOLOEVPDetectPredictor.setup_model`（32–40）

`super` 后 `done_warmup=True` 跳过 VP 场景 warmup。

### `set_prompts`（42–49）

保存 `self.prompts` dict（含 cls/bboxes 或 masks）。

### `is_per_image`（51–57，静态）

bboxes、cls 均为 ndarray 的 list 则为每图独立提示。

### `preprocess`（59–64）

`super.preprocess`；非 tile 时 `self.visuals = _prompts_to_tensor(imgs, im)`。

### `_batch_tiles`（66–76）

tile 模式：先对整图 batch 算 `embeddings`，逐 tile yield 时按 context 索引切片 `self.visuals`。

### `_prompts_to_tensor`（78–95）

目标 shape 为 letterbox 后 `imgs` 空间；tensor 源用同一 shape 列表；平铺或逐图调用 `_process_single_image`；`pad_sequence` → device，fp16/float。

### `_process_single_image`（97–117）

有 bbox：按 gain 缩放并加 padding 偏移；有 mask：`pre_transform` 堆叠，114 置 0；否则 ValueError；`LoadVisualPrompt().get_visuals(...)`。

### `inference`（119–121）

`super.inference(im, vpe=self.visuals)`。

### `get_vpe`（123–130）

`setup_source`；断言单图；预处理；`_prompts_to_tensor`；`model(..., return_vpe=True)` 返回嵌入。

### `YOLOEVPSegPredictor`（133–134）

空类：MRO 合并 VP 与 `SegmentationPredictor.postprocess/construct_result`。