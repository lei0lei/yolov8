# `ultralytics/models/yolo/yoloe/train_seg.py`

纯 Mixin 文件：通过多重继承把 `train.py` 中 YOLOE 训练变体接到 `SegmentationTrainer` 上，无新增方法体（除 `get_model` 别名）。

## 逐段说明

### `YOLOESegTrainer`（8–9）

`YOLOETrainer` + `SegmentationTrainer`：检测侧 YOLOE 逻辑与分割 Trainer 初始化顺序由 MRO 解析。

### `YOLOEPESegTrainer`（12–15）

基类仅用 `SegmentationTrainer`；`get_model = YOLOEPETrainer.get_model` 共享线性探测建模范式。

### `YOLOESegTrainerFromScratch`（18–19）

`YOLOETrainerFromScratch` + `YOLOESegTrainer`。

### `YOLOESegVPTrainer`（22–23）

`YOLOEVPTrainer` + `YOLOESegTrainerFromScratch`：从零训练 + 视觉提示增广。