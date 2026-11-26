# `ultralytics/solutions/trackzone.py`

`TrackZone` 用凸包定义 ROI，`bitwise_and` 掩膜后仅在区内做跟踪与标注，区外像素不参与推理。

## 逐段说明

### `__init__`（第 38～47 行）

默认四边形或用户 `region` 做 `cv2.convexHull`；`mask=None`。

### `process`（第 49～88 行）

尺寸变化时重建 `fillPoly` 掩膜；对掩膜帧 `extract_tracks`；画白色区界；画区内检测框；返回 `total_tracks`。