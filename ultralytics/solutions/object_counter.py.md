# `ultralytics/solutions/object_counter.py`

`ObjectCounter` 在配置区域（线段或多边形）上对跟踪目标做进/出计数：依据相邻帧质心穿越线段或与区域几何关系判断方向，每 track_id 只计一次，并按类别累计 `classwise_count`。

## 逐段说明

### `__init__`（第 40～52 行）

初始化 in/out 计数、`counted_ids`、`classwise_count`、`show_in`/`show_out`、`margin`。

### `count_objects`（第 54～121 行）

已计数或无 `prev_position` 则返回；**两点多边形**视为线段：segment 与区域线相交则按区域主方向（竖直比 x、水平比 y）判 IN/OUT；**多点区域**：当前点在区内或从区外穿入的 segment，按区域宽高比判左右/上下运动为 IN 或 OUT；成功后 `counted_ids.add`。

### `forget_tracks`（第 123～130 行）

从 `counted_ids` 移除退役 ID。

### `display_counts`（第 132～150 行）

汇总非零 IN/OUT 类标签，`display_analytics` 右上角绘制。

### `process`（第 152～205 行）

懒初始化 region；跟踪、画区、逐目标画框、存历史、`count_objects`；返回 `SolutionResults` 含 in/out/classwise/total_tracks。