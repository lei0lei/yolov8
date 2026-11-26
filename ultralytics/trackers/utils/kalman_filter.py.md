# `ultralytics/trackers/utils/kalman_filter.py`

实现 BYTE/OC 用的 `KalmanFilterXYAH`（状态 x,y,a,h 及速度）与 BoT/TrackTrack 用的 `KalmanFilterXYWH`（x,y,w,h）。均为常速度模型：8 维 mean、`_motion_mat` 积分、`predict` 加过程噪声，`update` 经 `project` 求卡尔曼增益；可选 `confidence` 做 NSA 测量噪声缩放。`gating_distance` 供门控（马氏/欧氏）。

## 逐段说明

### 类 `KalmanFilterXYAH.__init__`（第 37～55 行）

4 维观测×2 速度，`motion_mat` 单位块加 dt 耦合；`update_mat` 取前 4 维；位置/速度噪声权重 1/20 与 1/160。

### `initiate`（第 57～89 行）

观测拼零速度；对角协方差 std 与高度 h 成比例（aspect 维固定小方差）。

### `predict`（第 91～126 行）

由 mean[3]（h）缩放 std_pos/std_vel，加 motion_cov，`mean @ motion_mat.T` 与协方差 sandwich。

### `project`（第 128～159 行）

取 mean/cov 前 4 维，加 innovation 对角阵；`confidence` 时放大 innovation。

### `multi_predict`（第 161～199 行）

批量版 predict，逐行 std 用 `mean[:,3]`。

### `update`（第 201～232 行）

解 `projected_cov` 得增益，创新 `measurement - projected_mean`，更新 mean/cov。

### `gating_distance`（第 234～286 行）

`project` 后可仅位置 2 维；gaussian 为平方欧氏，maha 为 Cholesky 马氏距离。

### 类 `KalmanFilterXYWH`（第 289～485 行）

`initiate`/`predict`/`project`/`multi_predict` 的 std 用 w、h 分别缩放（与 XYAH 公式平行）；`update` 委托父类并传入 `confidence`。