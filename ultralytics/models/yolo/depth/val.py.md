# `ultralytics/models/yolo/depth/val.py`

`DepthValidator` 继承 `DetectionValidator` 但剥离检测逻辑：无 NMS、无框 mAP，`DepthMetrics` 聚合 δ1/abs_rel/rmse/silog；DDP 用 all_reduce 累加器而非检测 stats。

## 逐段说明

### `__init__`（25–34）

`super` 后 `task=depth`。

### `init_metrics`（36–39）

`DepthMetrics(max_depth=data 或 100)`，`clear_stats()`。

### `preprocess_targets`（41–45）

`super` 后 `batch["depth"].float()`。

### `postprocess`（47–49）

原样返回 preds，不做 NMS。

### `update_metrics`（51–60）

GT/pred 对齐通道维；空间尺寸不一则 `interpolate_bilinear`；`metrics.update_stats(preds, gt_depth)`。

### `get_stats`（62–69）

`metrics.process()` 返回 `results_dict`（跨 rank 和并在 `gather_stats` 完成）。

### `gather_stats`（71–90）

无 DDP 则 return；`_totals` 与 `_count` 张量 `all_reduce SUM`；写回 metrics。

### `print_results`（92–111）

表行 `depth_val` + 图像数 + 四项指标。

### `finalize_metrics`（113–116）

写 `speed`、`save_dir`。

### `get_desc`（118–120）

列名 Class/Images/delta1/abs_rel/rmse/silog。

### `plot_predictions`（122–135）

`plot_images` 标签键 `depth`，热图式可视化。