# `examples/python/Interactive-Tracking/main.py`

本示例在摄像头画面上用 YOLO 跟踪全部目标，并支持鼠标左键点选某一 track ID 做“聚焦跟踪”：为选中框绘制十字瞄准线、脉动圆心与 `ACTIVE` 高亮标签，其余目标以虚线框显示。可通过 `c` 清除选择、`q` 退出。模块顶层配置模型、阈值、tracker 与可选 GPU/录屏；主循环在文件末尾而非 `main()` 函数中。

## 逐段说明

### 模块级配置（第 13–53 行）

`enable_gpu`、`model_file`、`conf/iou/max_det`、`tracker` 与 `track_args`；全局状态 `selected_object_id`、`latest_detections` 等。

### `get_center`（第 56–68 行）

输入 xyxy 整数坐标，返回中心点 `( (x1+x2)//2, (y1+y2)//2 )`。

### `extend_line_from_edge`（第 71–93 行）

从框边中点沿 `up/down/left/right` 延伸到图像边界，供瞄准线终点计算。

### `draw_tracking_scope`（第 96–112 行）

对 bbox 四边中点各画一条线到对应方向图像边缘。

### `click_event`（第 115–144 行）

`EVENT_LBUTTONDOWN` 时在 `latest_detections` 中找包含点击点的最小面积框，解析 track_id 与类名并记日志。

### 主循环（第 147–239 行）

注册窗口与鼠标回调；`model.track` 更新 `latest_detections`；绘制全部轨迹（选中与非选中样式分支）；可选 FPS 叠字；懒初始化 `VideoWriter`；`q` 退出/`c` 重置选择；释放资源。