# `ultralytics/solutions/instance_segmentation.py`

对分割模型输出做实例可视化：默认 `yolo26n-seg.pt`，在 `extract_tracks` 后取 masks，用 `Results.plot` 实例着色绘制框与 mask。

## 逐段说明

### `__init__`（第 37～49 行）

设置默认分割权重；读取 show_conf/labels/boxes。

### `process`（第 51～87 行）

`extract_tracks`；无 masks 警告并原图返回；否则组装 `Results` 调 `plot(..., color_mode="instance")`；`display_output`；返回 `total_tracks`。