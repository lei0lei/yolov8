# `tests/test_solutions.py`

本文件测试 Ultralytics Solutions 套件（不含 DistanceCalculation 与安防告警）：对各 solution 类跑短视频、覆盖 OBB 角点、Analytics 图表、Parking JSON、相似度搜索 Streamlit、OCR 行分组等边界。树莓派与 torch<2.4 对部分用例 skip；大矩阵 `test_solution` 在树莓派整段 skip。

## 逐段说明

### `process_video`（第 27–42 行）

`VideoCapture` 逐帧调用 `solution(im[, frame_count])`，断言能打开视频。

### `test_solution`（第 45–217 行）

参数化约二十种 solution 配置；解析 `temp_*` 路径、parking 资产、`solution_assets` 视频；Analytics 需 `needs_frame_count`；Streamlit 用 `AppTest` 无异常；其余 `process_video`。

### `test_left_click_selection`（第 220–226 行）

`DistanceCalculation` 三次点击应选中 track 1 与 3（重叠取最小框）。

### `test_left_click_selection_obb`（第 229–235 行）

OBB 四角点在点击内应选中 track 1。

### `test_solution_obb_boxes`（第 238–265 行）

五类 solution 在 parking OBB 首帧 `plot_im` 非空。

### `test_object_blurrer_obb_outside_frame`（第 268–277 行）

完全出画 OBB 不导致 `cv2.blur` 崩溃。

### `test_right_click_reset`（第 280–285 行）

右键清空 `selected_boxes`。

### `test_parking_json_none`（第 288–295 行）

`json_path=None` 抛 ValueError 则 skip。

### `test_analytics_graph_not_supported`（第 298–305 行）

非法 `analytics_type` 抛含 `Unsupported analytics_type`。

### `test_area_chart_padding`（第 308–324 行）

area 图多类折线历史与 `max_points` 截断行为。

### `test_config_update_method_with_invalid_argument`（第 327–334 行）

`SolutionConfig.update` 非法键 `ValueError`。

### `test_plot_with_no_masks`（第 337–342 行）

空帧实例分割仍返回 `plot_im`。

### `test_similarity_search` / `test_similarity_search_complete`（第 345–367 行）

`SearchApp` 与 `VisualAISearch` 端到端查询非空。

### `test_distance_calculation_process_method`（第 370–387 行）

mock 跟踪后 `process` 返回正 `pixels_distance`。

### `test_object_crop_with_show_True`（第 390–392 行）

`ObjectCropper(show=True)` 初始化覆盖告警路径。

### `test_display_output_method`（第 395–406 行）

mock 验证 `imshow`/`waitKey`/`destroyAllWindows` 调用。

### `test_ocr_lines`（第 409–418 行）

`OCR.lines` 按 y 分行并左右排序为 `[[1,2],[0]]`。