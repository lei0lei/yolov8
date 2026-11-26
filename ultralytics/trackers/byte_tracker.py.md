# `ultralytics/trackers/byte_tracker.py`

本文件实现 ByteTrack 核心：`STrack` 单轨迹（卡尔曼 XYAH、框格式转换、激活/更新/重激活）与 `BYTETracker` 多阶段数据关联。每帧将检测按高低分切分，先联合 tracked+lost 池做高分匹配，再对剩余 tracked 与低分检测 IoU 匹配，未确认轨迹单独关联，最后初始化新轨迹并清理超时 lost。匹配代价默认为 IoU（可 fuse_score），可选 GMC 在关联前 warp 状态。

## 逐段说明

### 类 `STrack` — `__init__`（第 56～79 行）

解析 `xywh(+angle)+idx`；OBB 长边规范化；`_tlwh` 由 `xywh2ltwh` 得到；`mean/covariance` 待激活；记录 score、cls、idx、angle。

### `STrack.predict`（第 81～86 行）

非 `Tracked` 时将速度分量 mean[7] 置 0 再 `kalman_filter.predict`。

### `STrack.multi_predict`（第 88～101 行）

批量 mean/cov，非 Tracked 行清零 vh，共享 `shared_kalman.multi_predict`。

### `STrack.activate`（第 103～114 行）

分配 `track_id`，`initiate(convert_coords(_tlwh))`；`Tracked` 状态；首帧 `is_activated=True`。

### `STrack.re_activate` / `update`（第 116～161 行）

`kalman_filter.update` 用新观测；更新 score/cls/idx；OBB 角度按 π 周期对齐防翻转。

### 坐标属性与 `result`（第 163～215 行）

`tlwh`/`xyxy`/`xywh`/`xywha` 从 mean 或 `_tlwh` 导出；`result` 拼接 box、track_id、score、cls、idx。

### 类 `BYTETracker.__init__`（第 257～272 行）

初始化三轨迹池、`frame_id`、`max_frames_lost=track_buffer`、卡尔曼、`reset_id`。

### `BYTETracker.update` 主流程（第 274～321 行）

`frame_id++`；`_split_detections` 得 high/low 与 mask；`init_track` 构建 STrack 并修正 idx；分 unconfirmed/tracked；`joint_stracks`+`multi_predict`；GMC 钩子；**一阶段** `get_dists`+`linear_assignment`+`_apply_matches`；OCR 钩子默认透传；**二阶段** 仅 Tracked 未匹配 vs 低分 IoU 0.5，其余 `mark_lost`；**未确认** IoU 0.7 匹配或 `mark_removed`；**新轨迹** 分数 ≥ `new_track_thresh` 则 `activate`；**移除** lost 超 `max_frames_lost`；`merge_track_pools`；`_format_output` 仅 `is_activated` tracked。

### 辅助方法要点

- `_split_detections`（323～339）：剔除零宽高，按 `track_high_thresh`/`track_low_thresh` 划分。
- `_first_association`（388～399）：IoU(+fuse) 与 `match_thresh` 匈牙利匹配。
- `_apply_match`（429～436）：Tracked→`update`，否则 `re_activate` 入 refind。
- `_second_association`（438～460）：低分 Byte 第二阶段。
- `_unconfirmed_association`（462～488）：未确认轨迹 0.7 阈值。
- `_init_new_tracks`（490～503）：新高分检测 activate。
- `get_dists`（527～532）：`iou_distance`，可选 `fuse_score`。