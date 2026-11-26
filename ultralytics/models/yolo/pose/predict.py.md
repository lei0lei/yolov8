# `ultralytics/models/yolo/pose/predict.py`

`PosePredictor` 继承 `DetectionPredictor`：检测框后处理不变，在 `construct_result` 中追加关键点并 `scale_coords` 到原图。

## 逐段说明

### `__init__`（30–41）

`self.args.task = "pose"`。

### `construct_result`（43–66）

- `super().construct_result` 得带 boxes 的 `Results`。
- `pred[:,6:]` reshape 为 `(N, *model.kpt_shape)`。
- `ops.scale_coords` 将关键点从输入图尺寸映射到 `orig_img`。
- `result.update(keypoints=pred_kpts)` 写回。