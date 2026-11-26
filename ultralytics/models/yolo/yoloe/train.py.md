# `ultralytics/models/yolo/yoloe/train.py`

YOLOE 训练族：标准 `YOLOETrainer`、线性探测 `YOLOEPETrainer`、从零 `YOLOETrainerFromScratch`、无 prompt `YOLOEPEFreeTrainer`、视觉提示 `YOLOEVPTrainer`；均扩展检测/世界训练管线。

## 逐段说明

### `YOLOETrainer.__init__`（37–49）

禁止 compile；`overlap_mask=False`（相对分割默认）。

### `YOLOETrainer.get_model`（51–79）

`YOLOESegModel` 或 `YOLOEModel`；`nc=min(data["nc"],80)`；load weights。

### `get_validator` / `build_dataset`（81–100）

YOLOE 专用 validator；`multi_modal=train`。

### `get_class_counts`（102）

`BaseTrainer.get_class_counts`。

### `YOLOEPETrainer.get_model`（116–156）

删 `savpe`；须 weights；eval→`set_classes`+`fuse` 或 `get_vocab`；仅 cv3/one2one_cv3 末 conv 可训练 deepcopy。

### `YOLOEPETrainer.get_validator`（158–161）

普通 Detection/SegmentationValidator（无 YOLOE prompt 要求）。

### `YOLOETrainerFromScratch.build_dataset`（175–189）

委托 `WorldTrainerFromScratch.build_dataset`。

### `YOLOEPEFreeTrainer`（203–213）

`preprocess_batch` 不加 txt_feats；`set_text_embeddings` 空实现。

### `YOLOEVPTrainer.build_dataset`（226–244）

`super` 后对 transforms 追加 `LoadVisualPrompt`。

### `_close_dataloader_mosaic`（246–253）

关 mosaic 后再追加 `LoadVisualPrompt`（concat 或单集）。