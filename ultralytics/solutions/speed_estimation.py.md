# `ultralytics/solutions/speed_estimation.py`

`SpeedEstimator` 对每个 track_id 存 `max_hist` 个质心；满历史后按像素位移、`fps` 与 `meter_per_pixel` 算 km/h，锁定后不再更新，标签显示速度。

## 逐段说明

### `__init__`（第 43～59 行）

读 fps、max_hist、meter_per_pixel、max_speed；初始化字典与 `locked_ids`。

### `forget_tracks`（第 61～68 行）

清除退役 ID 的速度相关状态。

### `process`（第 70～126 行）

`frame_count+=1`；跟踪；`store_tracking_history`；新 ID 初始化 deque 与起始帧；未锁定则 append 质心；`len(trk_hist)==max_hist` 时算距离/时间得速度并锁定、释放历史；已锁定者在框上显示 km/h；`speed_dict` 为当前可见且已锁定 ID 的子集。