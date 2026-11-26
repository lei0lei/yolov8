# `ultralytics/solutions/queue_management.py`

`QueueManager` 在多边形区域内统计当前帧目标数量（质心在内即计入），并在区域中心显示队列人数。

## 逐段说明

### `__init__`（第 36～42 行）

`initialize_region`，`counts=0`，记录 `region_length`。

### `process`（第 44～82 行）

每帧 `counts` 归零；`extract_tracks`；画区域；逐目标画框，≥3 点区域且质心在 `r_s` 内则 `counts+=1`；`queue_counts_display`；返回 `queue_count`。