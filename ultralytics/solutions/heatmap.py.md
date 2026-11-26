# `ultralytics/solutions/heatmap.py`

`Heatmap` 继承 `ObjectCounter`，在目标框内累积径向热力权重，归一化后 `applyColorMap` 与原图 `addWeighted` 叠加；若配置 `region` 则同时执行穿线计数逻辑。

## 逐段说明

### `__init__`（第 38～51 行）

有 region 则 `initialize_region`；保存 `colormap`，`heatmap=None`。

### `heatmap_effect`（第 53～74 行）

外接框裁剪到图像；在 ROI 网格上对距中心≤半径的点 `heatmap[...]+=2`。

### `process`（第 76～129 行）

懒创建与帧同尺寸的 `heatmap`；跟踪后对每框 `heatmap_effect`；有 region 时 `store_tracking_history`+`count_objects`；可选 `display_counts`；热力非零则归一化上色 0.5 融合；返回与 `ObjectCounter` 相同计数字段。