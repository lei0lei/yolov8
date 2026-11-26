# `ultralytics/trackers/oc_sort.py`

OC-SORT 在 ByteTrack 上增加观测中心机制：`OCSortTrack` 保存真实检测 xyxy 历史、速度方向与 Kalman 快照；关联代价含 IoU(+分数融合)+惯性加权 OCM 角差；一阶段后 `_post_first_association` 对未匹配轨迹做 OCR（用 last observation 算 IoU，丢失轨迹可 `apply_oru` 填遮挡隙）。`use_byte=False` 时跳过低分二阶段，直接把未匹配 Tracked 标 lost。

## 逐段说明

### 类 `OCSortTrack`

- `__init__`（29～44）：`observations` 字典、`velocity`、`delta_t`、保存 mean 副本。
- `activate`/`update`/`re_activate`：`_record_observation` 后调父类，刷新 `_saved_mean` 与 `_compute_velocity`。
- `_record_observation`（90～101）：保留最近 `delta_t+2` 帧观测。
- `_compute_velocity`（103～134）：当前中心与至少 `delta_t` 帧前中心差分归一化。
- `apply_oru`（136～167）：按空隙在 last_obs 与新观测间线性插值虚拟框，逐步 predict+update 修复 Kalman。

### 类 `OCSORT`

- `__init__`（186～196）：`delta_t`、`inertia`、`use_byte`。
- `get_dists`（215～220）：IoU、可选 fuse_score、加 `inertia * _velocity_direction_cost`、`_fuse_appearance` 钩子。
- `_ocr_associate`（222～249）：`_ocr_distance`+fuse+外观，匹配时 Lost 先 `apply_oru` 再 `re_activate`。
- `_post_first_association`（251～278）：先 Tracked 未匹配 OCR，再 Lost 对剩余检测 OCR。
- `_second_association`（280～297）：`use_byte` 为真才调父类低分关联，否则全部未匹配 Tracked→lost。
- `_velocity_direction_cost`（299～334）：检测中心相对历史观测方向与 track.velocity 夹角/π。
- `_ocr_distance`（336～358）：轴对齐用 `last_observation`，OBB 退化为预测 `xywha` IoU。