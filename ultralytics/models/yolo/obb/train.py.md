# `ultralytics/models/yolo/obb/train.py`

`OBBTrainer` 继承 `DetectionTrainer`：强制 `task=obb`，模型换 `OBBModel`，验证换 `OBBValidator`；数据与多尺度与检测相同。

## 逐段说明

### `__init__`（34–47）

`overrides["task"]="obb"`。

### `get_model`（49–74）

`set_model_names_for_load(OBBModel(...))`；可选 `load(weights)`。

### `get_validator`（76–80）

`OBBValidator`。