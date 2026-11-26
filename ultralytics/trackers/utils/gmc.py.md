# `ultralytics/trackers/utils/gmc.py`

`GMC`（Generalized Motion Compensation）估计相邻帧 2×3 仿射，供 `multi_gmc` 对齐卡尔曼状态。支持 `sparseOptFlow`（默认）、`ecc`、`orb`、`sift` 或恒等变换；可下采样加速，特征法可屏蔽检测框内关键点。

## 逐段说明

### `__init__`（第 45～95 行）

保存 `method`/`downscale`；ORB/SIFT 初始化 detector+matcher；ECC 设迭代与 EUCLIDEAN 模式；sparseOptFlow 设角点参数；首帧标志 false。

### `apply`（第 97～122 行）

按 method 分发 `apply_features`/`apply_ecc`/`apply_sparseoptflow`，未知则 `eye(2,3)`。

### `apply_ecc`（第 124～163 行）

转灰度、可选下采样；首帧缓存并返回单位矩阵；`findTransformECC` 求 H，尺度还原平移；失败警告后仍返回；更新 `prevFrame`。

### `apply_features`（第 165 行起）

检测+匹配前后帧描述子，RANSAC 估计仿射（详见源码内 ORB/SIFT 分支与检测掩膜逻辑）。

### `apply_sparseoptflow`（第 288 行起）

Lucas-Kanade 光流跟踪角点，最小二乘拟合变换矩阵。

### `reset_params`（第 350 行起）

清空前一帧与初始化标志，切换视频时调用。