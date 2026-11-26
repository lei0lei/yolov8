# `ultralytics/trackers/__init__.py`

本包对外导出 Ultralytics 支持的多目标跟踪算法类及预测器注册函数。从 `bot_sort`、`byte_tracker`、`deep_oc_sort`、`fast_tracker`、`oc_sort` 子模块分别导入 `BOTSORT`、`BYTETracker`、`DeepOCSORT`、`FASTTracker`、`OCSORT`；从 `track` 导入 `register_tracker`，从 `track_tracker` 导入 `TRACKTRACK`。`__all__` 列出上述符号，便于 `from ultralytics.trackers import BYTETracker` 等写法，与 `yolo track tracker=*.yaml` 配置中的 `tracker_type` 对应。

## 逐段说明

### 导入与导出（第 3～19 行）

- 第 3～9 行：相对导入七个跟踪实现与注册入口。
- 第 11～19 行：`__all__` 元组限定公开 API，注释说明允许更简洁的 import。