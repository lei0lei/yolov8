# `ultralytics/trackers/fast_tracker.py`

FastTracker 扩展 ByteTrack：`FastSTrack` 维护 mean 历史环缓冲与遮挡标记；未匹配且被其它活跃轨迹高覆盖率的 track 判定遮挡，回滚速度与位置、放大高度、阻尼速度，并延长 re-find 窗口；新检测与活跃框 IoU 过高则抑制建新 ID；输出仅含本帧更新的轨迹。

## 逐段说明

### 类 `FastSTrack`

- `__init__`（45～60）：`deque(maxlen=history_len)` 与遮挡计数器。
- `_push_history`（62～65）：复制 mean/cov 入队。
- `activate`/`re_activate`/`update`（67～101）：父类逻辑后 `_push_history`；重激活清空遮挡状态。

### 类 `FASTTracker`

- `__init__`（155～177）：读取遮挡/抑制相关阈值；`_history_len` 与回滚深度对齐。
- `init_track`（179～196）：`FastSTrack(..., history_len=self._history_len)`。
- `_apply_match`（198～203）：成功匹配清除遮挡计数。
- `_second_association`（205～222）：低分关联后 `_handle_occlusions` 替代原 mark-lost 循环。
- `_init_new_tracks`（224～252）：`bbox_ioa` 与活跃池≥`init_iou_suppress` 则跳过 activate，否则激活并扩充 active_stack。
- `_remove_stale_lost`（254～262）：`was_recently_occluded` 且在 `occ_reappear_window` 内可免于移除。
- `_format_output`（264～268）：要求 `frame_id == self.frame_id`。
- `_handle_occlusions`（271～349）：向量化 `bbox_ioa` 覆盖率；超 `occ_cover_thresh` 首次进入遮挡并从历史回滚速度/位置、放大 h、阻尼速度；`not_matched>2` 且非遮挡或超过 `active_occ_to_lost_thresh` 则 `mark_lost`。