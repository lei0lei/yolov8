# `ultralytics/models/yolo/obb/predict.py`

`OBBPredictor` 继承 `DetectionPredictor`，仅改 `task=obb` 与 `construct_result`：NMS 侧已在父类 `rotated=True`；结果写入 `Results.obb`（xywhr+conf+cls）。

## 逐段说明

### `__init__`（30–39）

`super` 后 `self.args.task = "obb"`，使 NMS 与后处理走旋转框分支。

### `construct_result`（41–58）

- `rboxes = cat([pred[:,:4], pred[:,-1:]])` 合框与角。
- `scale_boxes(..., xywh=True)` 映射到原图。
- `obb = cat([rboxes, pred[:,4:6]])` 含 conf、cls。
- `Results(..., obb=obb)`，无 axis-aligned `boxes`。