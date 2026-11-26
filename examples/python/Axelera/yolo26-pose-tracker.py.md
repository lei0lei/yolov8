# `examples/python/Axelera/yolo26-pose-tracker.py`

Axelera 上的 YOLO26 姿态示例：NMS-free 导出用自定义 `ConfidenceFilter` 代替 `decode_pose`；可选 `tracktrack/bytetrack` 等跟踪算子。渲染用 OpenCV 自绘 COCO 骨架与调色板，与 `yolo11-seg.py` 的 display.App 路径互补。支持摄像头/视频/单图；无头模式写 mp4。

## 逐段说明

### 常量（第 39–78 行）

`COCO_SKELETON`、`POSE_PALETTE`、`KPT_COLORS`/`LIMB_COLORS`、`GOLDEN_RATIO_CONJUGATE`。

### `get_track_color`（第 81–85 行）

按 track_id 用黄金比例分布在 HSV 上生成稳定 BGR 色。

### `ConfidenceFilter`（第 88–105 行）

`op.Operator`：去掉 batch 维，按第 4 列 score 阈值过滤 `(N,57)` 行。

### `build_pipeline`（第 108–126 行）

BGR→RGB、letterbox、tensor、load、ConfidenceFilter、`to_image_space` 关键点列；可选 `ax_pose`+`tracker`。

### `draw_pose`（第 129–165 行）

无跟踪：逐检测画框、COCO 骨架线与关键点（conf>0.5）。

### `draw_tracked_poses`（第 168–212 行）

跟踪对象：ID 色框、骨架与点；关键点从 `tracked.tracked` 读取。

### `main`（第 215–283 行）

解析 model/source/conf/tracker/no-display/output；`VideoCapture` 循环 `pipeline(frame)`；有跟踪与无跟踪分支绘图；头模式 `imshow` 或写 `VideoWriter`。