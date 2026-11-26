# `ultralytics/solutions/parking_management.py`

`ParkingPtsSelection` 为 Tkinter 工具：点击四点定义车位多边形，保存至 `bounding_boxes.json`（按画布缩放还原原图坐标）。`ParkingManagement` 读 JSON 区域，质心落在多边形内且有待检测则标记占用，统计空余/占用并叠加分析条。

## 逐段说明

### 类 `ParkingPtsSelection`（第 16～171 行）

- `__init__`：tkinter 不可用则警告退出；画布按钮 Upload/Remove/Save。
- `upload_image`：EXIF 校正、等比缩放到画布，绑点击。
- `on_canvas_click`：红点，满 4 点 append `rg_data` 蓝框。
- `save_to_json`：坐标按 canvas/原图比例缩放写入 JSON。

### 类 `ParkingManagement`

- `__init__`（197～220）：必填 `json_file`，加载列表，`pr_info` 计数器，占用/空闲配色。
- `process`（222～280）：跟踪得质心；每 region `pointPolygonTest`≥0 则 occupied，`display_objects_labels` 类名；多边形颜色绿/红；更新 `Occupancy`/`Available` 分析条；返回 `filled_slots`/`available_slots`。