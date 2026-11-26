# `ultralytics/trackers/basetrack.py`

本模块定义多目标跟踪的公共枚举与各跟踪器共享的轨迹基类。`TrackState` 用整型常量表示轨迹生命周期：新建、已跟踪、丢失、已移除。`BaseTrack` 维护全局递增的 `track_id` 计数器及每条轨迹的状态、分数、起止帧号等字段；`activate`、`predict`、`update` 为子类必须实现的抽象钩子，`mark_lost`/`mark_removed` 仅改状态，`reset_id` 清零全局 ID。

## 逐段说明

### 类 `TrackState`（第 7～25 行）

- 第 22～25 行：类属性 `New=0`、`Tracked=1`、`Lost=2`、`Removed=3`，供各 tracker 判断池内轨迹所处阶段。

### 类 `BaseTrack`（第 28～102 行）

#### `BaseTrack` 类属性与 `__init__`（第 57～66 行）

- 第 57 行：类级 `_count = 0` 作为全局 ID 源。
- 第 61～66 行：实例 `track_id=0`、`is_activated=False`、`state=New`、`score=0`、`start_frame`/`frame_id=0`。

#### 属性 `end_frame`（第 68～71 行）

返回 `frame_id`，表示该轨迹最近一次被更新的帧号。

#### 静态方法 `next_id`（第 73～77 行）

`BaseTrack._count += 1` 并返回新 ID。

#### `activate` / `predict` / `update`（第 79～89 行）

各自 `raise NotImplementedError`，由 `STrack` 等子类实现卡尔曼 与观测融合。

#### `mark_lost`（第 91～93 行）

`self.state = TrackState.Lost`。

#### `mark_removed`（第 95～97 行）

`self.state = TrackState.Removed`。

#### 静态方法 `reset_id`（第 99～102 行）

`BaseTrack._count = 0`，新视频或 `tracker.reset()` 时调用。