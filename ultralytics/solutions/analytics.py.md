# `ultralytics/solutions/analytics.py`

`Analytics` 在跟踪结果上绘制实时统计图（折线/柱状/饼/面积），matplotlib Agg 后端渲染为图像帧。每 `update_every` 帧刷新一次，其余帧复用 `last_plot_im` 缓存。

## 逐段说明

### `Analytics.__init__`（第 53～96 行）

读 `analytics_type`、标签名、配色与 `figsize`；折线/面积模式建 `Figure`+`line`；柱/饼建 `plt.subplots`，饼图 `axis("equal")`。

### `process`（第 98～141 行）

`extract_tracks`；折线：`total_counts=len(boxes)`，按帧号条件 `update_graph`；柱/饼/面积：`Counter` 类名计数后同样条件更新；非法 type 抛 `ValueError`；返回 `classwise_count` 与 `total_tracks`。

### `update_graph`（第 143～264 行）

- `count_dict is None`：追加折线 x/y，超 `max_points` 截断，设样式后绘制。
- `plot=="area"`：按标签维护多序列 y，clear 后 fill_between+plot 每类。
- `plot=="bar"`：clear，颜色映射，柱顶标数，bar.set_label。
- `plot=="pie"`：有总数才画 wedge，右侧 legend 带百分比。
- 公共：轴标题、网格、legend 前景色、`relim/autoscale_view`，`canvas.draw` 转 RGBA→BGR，可选 `display_output`。