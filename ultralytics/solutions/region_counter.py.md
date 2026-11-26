# `ultralytics/solutions/region_counter.py`

`RegionCounter` 支持多个命名多边形区域：质心落在区域内则该区域瞬时计数+1，帧末在区域质心画 `adaptive_label` 显示计数后清零区域计数器（`region_counts` 字典保留本帧快照）。

## 逐段说明

### `__init__`（第 40～53 行）

`region_template` 默认字段；`initialize_regions()`。

### `add_region`（第 55～94 行）

少于 3 点抛错；建 `Polygon`+`prepared_polygon`，追加 `counting_regions`。

### `initialize_regions`（第 96～107 行）

无 region 则默认矩形；dict 或单列表统一为 name→points。

### `process`（第 109～149 行）

跟踪；重置 `region_counts`；质心 `prepared_polygon.contains` 则对应 region `counts++`；画区与计数标签；每 region `counts` 复位 0；返回 `region_counts`。