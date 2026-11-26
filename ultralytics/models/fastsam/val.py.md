# `ultralytics/models/fastsam/val.py`

`FastSAMValidator` 在 YOLO 分割验证器基础上固定任务为 `segment` 并关闭绘图，避免 FastSAM 验证时 ConfusionMatrix 等分割专用图触发不兼容路径。数据加载、mAP 与 mask 指标仍走 `SegmentationValidator` 父类实现。

## 逐段说明

### 行 5–6：导入

从 `ultralytics.models.yolo.segment` 导入 `SegmentationValidator`。

### `FastSAMValidator`（行 8）

继承 `SegmentationValidator`；文档说明扩展目的及禁用 plots 的动机。

### `__init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None)`（行 26）

- **行 38**：调用 `super().__init__(...)` 初始化基类验证器（loader、指标、回调等）。
- **行 39**：将 `self.args.task` 设为 `"segment"`，与 FastSAM 门面一致。
- **行 40**：将 `self.args.plots` 设为 `False`，关闭混淆矩阵等绘图以避免错误。