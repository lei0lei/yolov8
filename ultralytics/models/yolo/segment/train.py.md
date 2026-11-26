# `ultralytics/models/yolo/segment/train.py`

`SegmentationTrainer` 继承 `DetectionTrainer`：`task=segment`，`SegmentationModel`，其余数据管线与检测共享。

## 逐段说明

### `__init__`（30–41）

`overrides["task"]="segment"`。

### `get_model`（43–64）

`SegmentationModel` + `set_model_names_for_load` + load。

### `get_validator`（66–70）

`SegmentationValidator`。