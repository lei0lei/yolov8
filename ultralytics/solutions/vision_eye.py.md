# `ultralytics/solutions/vision_eye.py`

`VisionEye` 从配置的 `vision_point`（观察点）向每个目标外接框中心画连线，模拟“视觉眼”透视关系。

## 逐段说明

### `__init__`（第 28～36 行）

`vision_point` 转整数元组。

### `process`（第 38～67 行）

`extract_tracks`；每目标 `box_label`+`visioneye(enclosing_box, vision_point)`；返回 `total_tracks`。