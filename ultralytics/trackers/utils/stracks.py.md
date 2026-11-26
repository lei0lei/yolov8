# `ultralytics/trackers/utils/stracks.py`

本模块为各 tracker 提供轨迹列表池化、检测框解析、GMC 批量变换与重复轨迹剔除等无状态工具函数。`merge_track_pools` 在每帧 `update` 末尾合并 activated/refind、维护 lost/removed 池并 IoU 去重。`parse_bboxes` 从 Results-like 对象拼接检测索引列。`joint_stracks`/`sub_stracks` 按 `track_id` 合并或差集。`remove_duplicate_stracks` 对 tracked 与 lost 池 IoU 过高的一对删除存活时间较短者。`multi_gmc` 用 8×8 块对角旋转矩阵 warp 卡尔曼均值与协方差（OBB 时尺度用 `norm(R[:,0])`）。

## 逐段说明

### `merge_track_pools`（第 18～56 行）

- 保留 `state==Tracked` 的旧 tracked，再 joint activated、refind。
- lost 池减去已 tracked 与 removed 的 ID，extend 本帧新 lost。
- `remove_duplicate_stracks` 去重 tracked/lost。
- `removed_stracks_frame = removed`，追加历史 removed 并截断至 `removed_buffer`。

### `parse_bboxes`（第 59～70 行）

选 `xywhr` 或 `xywh`，`concat` 列向量 `arange(N)` 作为检测 idx。

### `joint_stracks`（第 73～88 行）

`atracks` 的 ID 集合优先，从 `btracks` 追加未见 ID。

### `sub_stracks`（第 91～106 行）

过滤掉 `track_id` 出现在 `btracks` 中的项。

### `remove_duplicate_stracks`（第 109～142 行）

`pdist < dup_thresh` 的配对比较 `frame_id - start_frame`，删除寿命较短索引。

### `multi_gmc`（第 145～180 行）

- 空列表直接返回。
- 构造 `R8x8`：OBB 时 2×2 块用 `norm(R[:,0])*I`，否则 `abs(R)`。
- `mean` 左乘 `R8x8` 后中心加平移 `H[:2,2]`；协方差 sandwich 变换。
- OBB 轨迹 `angle` 加 `arctan2(R[1,0], R[0,0])`。