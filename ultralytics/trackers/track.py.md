# `ultralytics/trackers/track.py`

本模块把多类 tracker 接入 YOLO 预测流水线：`TRACKER_MAP` 映射 YAML 中 `tracker_type` 到实现类；`on_predict_start` 在 `mode=track` 时校验任务、加载 tracker 配置、按需注册 Detect 层预钩子提取 ReID 特征并实例化 tracker；`on_predict_postprocess_end` 将后处理框送入 `tracker.update` 并回写带 `track_id` 的结果；`register_tracker` 以可替换的 partial 回调注册到模型。

## 逐段说明

### `TRACKER_MAP`（第 18～26 行）

字符串键 `bytetrack`/`botsort`/`tracktrack`/`fasttrack`/`ocsort`/`deepocsort` 对应类。

### 函数 `on_predict_start`（第 29～96 行）

- 非 track 模式直接返回。
- 任务不在 `detect,segment,pose,obb` 且为已知 TASKS 时 `ValueError`。
- 流式 `dataset.bs` 决定 tracker 数量；`persist` 且已有足够 tracker 则跳过重建。
- `check_yaml` + `YAML.load` 得配置，`cfg.device` 对齐 predictor。
- 非法 `tracker_type` 则 `AssertionError`。
- 清除 `_feats`、移除旧 hook、恢复被 TRACKTRACK 包过的 `postprocess`。
- ReID `with_reid` 且 `model=auto`：若非标准 Detect 头则改用 `yolo26n-cls.pt`；否则注册 `pre_hook` 在 track 模式缓存 Detect 输入特征。
- 按需新建 tracker 列表，初始化 `vid_path`。
- 若 tracker 类有 `setup_predictor`，调用之。

### 函数 `on_predict_postprocess_end`（第 99～142 行）

- 非 track 返回。
- 可选 `compute_frame_extras` 提供每帧额外输入（如 TRACKTRACK 的 `dets_del`）。
- 每路结果选对应 tracker；非 persist 且视频路径变化则 `tracker.reset()`。
- 取 OBB 或 boxes 的 numpy，`tracker.update(det, orig_img, feats=..., dets_del=...)`。
- 无轨迹且存在未激活 track 时清空该帧结果以防闪现。
- 否则按 `tracks[:,-1]` 索引过滤 result，用 `tracks[:,:-1]` 更新 boxes/obb 张量。

### 函数 `register_tracker`（第 145～169 行）

遍历两个事件，若已有同 `func` 的回调则替换为新的 `partial(..., persist=persist)`，否则 `add_callback`。