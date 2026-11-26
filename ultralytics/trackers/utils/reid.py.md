# `ultralytics/trackers/utils/reid.py`

为 BoT-SORT / Deep OC-SORT / TrackTrack 提供 ReID：`ReID` 类对 `.pt` 走 YOLO embed，其它格式走 `AutoBackend`；`build_encoder` 在 `model=auto` 时直通 backbone 特征；`smooth_feature` 做 L2 归一化 EMA。

## 逐段说明

### 类 `ReID.__init__`（第 28～74 行）

选设备；`.pt` 加载 YOLO 并 warmup embed；否则 `AutoBackend`，可选下载官方 reid onnx，解析输入 batch/imgsz。

### `ReID._crop_detections` / `_crops_to_tensor`（第 76～96 行）

xywh→裁剪 BGR patch；堆叠归一化 resize 到 `imgsz`，可选 half。

### `ReID.__call__`（第 98～136 行）

裁剪；无有效 crop 返回全 None；PT 路径 `predictor(valid_crops)`；否则分批 forward（静态 batch 填充末块）；按 valid 掩码还原与检测一一对应列表。

### `build_encoder`（第 139～161 行）

`with_reid=False`→None；`model=="auto"` 返回 numpy 列表化特征；否则 `ReID(model)`。

### `smooth_feature`（第 164～186 行）

零范数返回 `(None, smooth)`；否则归一化 feat；`smooth=None` 时两者均为 feat；否则 EMA 后再归一化 smooth。