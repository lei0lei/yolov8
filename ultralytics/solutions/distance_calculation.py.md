# `ultralytics/solutions/distance_calculation.py`

交互式测算两跟踪目标质心像素欧氏距离：鼠标左键最多选两个 track 框，右键清空；选中两框时每帧更新坐标并 `plot_distance_and_line`。

## 逐段说明

### `__init__`（第 34～39 行）

`selected_boxes` 字典与 `centroids` 列表。

### `mouse_event_for_distance`（第 41～67 行）

左键：未满两个时命中框内且未选过的 `track_id` 加入字典；右键：清空选择。

### `forget_tracks`（第 69～73 行）

移除已删除 track 的选择项。

### `process`（第 75～132 行）

跟踪并画框；若已选两 ID 则同步框坐标；两选全时算质心对距离、`plot_distance_and_line`；重置 `centroids`；`show` 时注册鼠标回调；返回 `pixels_distance` 与 `total_tracks`。