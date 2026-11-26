# `ultralytics/trackers/bot_sort.py`

BoT-SORT 在 ByteTrack 框架上改用 `KalmanFilterXYWH`、`BOTrack`（ReID 特征 EMA）与 GMC 相机运动补偿。距离矩阵在 IoU 基础上用 `proximity_thresh` 掩膜，若启用 ReID 则与嵌入距离取逐元素 min，并受 `appearance_thresh` 门控。

## 逐段说明

### 类 `BOTrack`

- `__init__`（53～69）：`alpha=0.9`，可选初始 `update_features`。
- `update_features`（71～75）：`smooth_feature` 更新 curr/smooth。
- `predict`（77～84）：非 Tracked 清零速度 mean[6:8]。
- `re_activate`/`update`（86～96）：有新特征则先平滑再调 `STrack` 逻辑。
- `tlwh`（98～105）：XYWH 状态下由 mean 四角减半宽高。
- `multi_predict`（107～121）：批量预测，非 Tracked 清零速度维。
- `convert_coords`/`tlwh_to_xywh`（123～132）：测量空间为中心宽高。

### 类 `BOTSORT`

- `__init__`（168～181）：`GMC(gmc_method)`，`build_encoder(with_reid, model, device)`。
- `get_kalmanfilter`（183～185）：返回 `KalmanFilterXYWH`。
- `init_track`（187～195）：有 encoder 且图像则对每个框抽特征构造 `BOTrack`。
- `get_dists`（197～210）：IoU 距离；`dists>1-proximity` 掩膜；fuse_score；嵌入距离/2 与掩膜、appearance 阈值，再 `minimum`。
- `multi_predict`（212～214）：`BOTrack.multi_predict`。
- `reset`（216～219）：`super.reset` + `gmc.reset_params`。