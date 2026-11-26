# `ultralytics/solutions/ai_gym.py`

`AIGym` 基于姿态跟踪做健身重复计数：按 track_id 维护角度、次数与阶段；用 `SolutionAnnotator` 计算指定关键点夹角，与 `up_angle`/`down_angle` 比较切换 up/down 并在从 up 到 down 时递增计数。

## 逐段说明

### `AIGym.__init__`（第 33～47 行）

默认 `yolo26n-pose.pt`；`states` 为 `defaultdict`；从 CFG 读角度阈值与 `kpts`。

### `forget_tracks`（第 49～53 行）

父类清理后 `states.pop(track_id)`。

### `process`（第 55～119 行）

`extract_tracks`；若有框则批量取 keypoints numpy；每目标估计角、画关键点；角&lt;`down_angle` 且原 stage 为 up 则 count++，stage=down；角&gt;`up_angle` 则 stage=up；可选绘制角度/次数/阶段；`SolutionResults` 按当前 `track_ids` 列表对齐返回 workout 字段。