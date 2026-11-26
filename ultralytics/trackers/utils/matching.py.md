# `ultralytics/trackers/utils/matching.py`

本模块提供跟踪数据关联用的代价矩阵构建与匈牙利式线性分配。优先通过 `lap.lapjv` 在代价阈值内求解分配，不可用时回退 `linear_sum_assignment`。`iou_distance` 将轨迹与检测的 IoU 转为代价 `1-IoU`，OBB 用 `batch_probiou`，轴对齐框用 `bbox_ioa`。`embedding_distance` 用归一化特征余弦距离，缺失特征行/列置最大距离 2.0。`fuse_score` 将 IoU 相似度与检测置信度相乘后再转回代价，强化高分检测的匹配优先级。

## 逐段说明

### 依赖 `lap`（第 7～15 行）

导入失败时 `check_requirements("lap>=0.5.12")` 后重试。

### 函数 `linear_assignment`（第 18～59 行）

- 空矩阵：返回空 matches，未匹配索引为全部行、列。
- `use_lap=True`：`lapjv(..., cost_limit=thresh)`，`x[i]>=0` 为匹配。
- 否则：`linear_sum_assignment(min(cost, thresh))`，仅保留 `cost <= thresh` 的配对。

### 函数 `iou_distance`（第 62～98 行）

- 若传入已是 `ndarray` 列表则直接作框；否则从 track 取 `xywha` 或 `xyxy`。
- 5 维框走 `batch_probiou`，4 维走 `bbox_ioa(..., iou=True)`。
- 返回 `1 - ious`。

### 函数 `embedding_distance`（第 101～142 行）

- 堆叠 `smooth_feat`/`curr_feat`，无特征用零向量占位。
- 余弦距离 `1 - dot/(norm product)`，裁剪为非负。
- `missing_t`/`missing_d` 整行或列设为 2.0。

### 函数 `fuse_score`（第 145～168 行）

- `iou_sim = 1 - cost_matrix`，与每列检测 `score` 广播相乘得 `fuse_sim`，返回 `1 - fuse_sim`。