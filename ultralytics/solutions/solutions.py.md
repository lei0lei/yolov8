# `ultralytics/solutions/solutions.py`

Solutions 基础设施：`BaseSolution` 加载 `SolutionConfig`、YOLO 与 shapely 几何，统一 `extract_tracks`（track 或 predict）、历史与区域；`SolutionAnnotator` 扩展绘图（区域、健身、队列、VisionEye 等）；`SolutionResults` 承载各方案输出字段。

## 逐段说明

### 类 `BaseSolution.__init__`（第 77～139 行）

`SolutionConfig().update(**kwargs)`→`CFG`；懒导入 shapely；默认 model；`track_add_args` 收集 iou/conf/device 等；CLI 无 source 下载 demo；`check_imshow`、`track_history`、双 `Profile`。

### `adjust_box_label`（第 141～157 行）

可选 track_id 前缀+类名；`show_conf` 追加分数；`show_labels` 为假返回 None。

### `extract_tracks`（第 159～185 行）

`tracker is None` 则 predict 否则 `model.track(persist=True)`；区分 OBB/boxes；填充 boxes/clss/track_ids/confs 或清空；track 时 `forget_tracks(removed_stracks_frame)`。

### `forget_tracks`（第 187～194 行）

`track_history.pop`。

### `store_tracking_history`（第 196～215 行）

追加以中心点坐标，最长 30。

### `get_enclosing_box`（第 217～236 行）

OBB 四角 min/max 得 xyxy，否则原样。

### `initialize_region`（第 238～244 行）

无 region 默认四边形；≥3 点 Polygon 否则 LineString→`r_s`。

### `display_output`（第 246～269 行）

`show` 且 `env_check` 时 imshow，q 关窗。

### `process`（第 271～272 行）

子类实现。

### `__call__`（第 274～297 行）

`profilers[1]` 包 process；写 `result.speed` 字典；verbose 时打帧统计日志。

### 类 `SolutionAnnotator`

- `__init__`（337～356）：转调 `Annotator`。
- `draw_region`（358～375）：`polylines`+顶点圆。
- `queue_counts_display`（377～420）：区域中心算矩形底+putText。
- `display_analytics`（422～455）：右上堆叠标签条。
- `_point_xy` / `estimate_pose_angle`（457～495）：张量/ndarray 归一化+缓存角度。
- `draw_specific_kpts`（497～536）：按 indices 顺序连线画圆，conf 过滤。
- `plot_workout_information`（538～569）：背景矩形+文本，返回字高。
- `plot_angle_and_count_and_stage`（571～602）：三段纵向 workout 标签。
- `plot_distance_and_line`（604～641）：左上角距离框+质心连线。
- `display_objects_labels`（643～689）：居中类标签底。
- `sweep_annotator`（691～729）：竖线+可选标签。
- `visioneye`（731～749）：观察点与框中心连线。
- `adaptive_label`（751～806）：圆/矩形底标签，circle 限 3 字符。

### 类 `SolutionResults`（第 809～872 行）

`__init__` 默认各统计字段为 0/空，`kwargs` 覆盖；`__str__` 跳过 `plot_im` 列非空属性。