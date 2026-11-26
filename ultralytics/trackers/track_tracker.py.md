# `ultralytics/trackers/track_tracker.py`

TrackTrack 实现：宽松 NMS 恢复检测（TAI）、HMIoU+ReID+置信度+角点角度多代价、递减阈值迭代互最近邻匹配。`TTSTrack` 扩展 `BOTrack` 含四角速度、分数历史、NSA-Kalman；`TRACKTRACK.update` 分 high/low/recovered 检测池，GMC+`multi_predict` 后主关联，再 unconfirmed、可选 lost 重绑、TAI 抑制.spawn。

## 逐段说明

### 模块级代价函数（第 30～134 行）

- `_hmiou_distance`：垂直 IoU×普通 IoU 得 HMIoU 代价。
- `_angle_distance`：仅 IoU>0.1 支持的配对，历史框与检测四角方向与 track.velocity 夹角。
- `_confidence_distance`：投影分数与检测分差的绝对值。
- `_iterative_associate`：互最近邻+阈值每轮减 `reduce_step`。
- `_track_aware_nms`：与活跃轨迹或更强检测 IoU 过高则禁止新轨迹。

### `attach_raw_preds_hook` / `compute_dets_del`（第 136～188 行）

包装 `postprocess` 保存 tight 与 loose NMS 结果供恢复删除框。

### 类 `TTSTrack`（第 210～334 行）

- 分数自适应 EMA `update_features`；`get_history_box` 按 `delta_t` 回看；`activate` 可能保持 New 直至 `min_track_len`；`update` 更新四角速度、NSA `kalman_filter.update(..., confidence=score)`。

### 类 `TRACKTRACK`（第 337～603 行）

- `__init__`：多权重、GMC、encoder、ID 迭代器。
- `setup_predictor`：detect/obb 时挂 raw preds hook。
- `_cost_matrix`：组合 HMIoU、ReID、置信、角度，无 IoU 支持置 1.0。
- `update` 流程：划分 high/low；encoder 抽特征建 `TTSTrack`；`dets_del` 恢复高置信；pool=tracked+lost；GMC+predict；主 ` _iterative_associate`→update/re_activate 或 mark_lost；unconfirmed 对 leftover high；可选 lost 对 leftover；`_track_aware_nms` 后 activate 新轨迹；`merge_track_pools`；输出本帧激活轨迹 `result`。
- `reset`：清空池与 GMC。