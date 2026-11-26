# `ultralytics/utils/callbacks/comet.py`

Comet.ml 训练跟踪：**`on_pretrain_routine_start`** 创建/续跑实验；**`on_train_epoch_end`** 训练 loss；**`on_fit_epoch_end`** 验证指标/lr/模型信息，并按保存周期上传模型、混淆矩阵、图像预测；**`on_train_end`** 最终模型/图/表/args；**`teardown`** `experiment.end()`。仅 `detect`/`segment` 支持验证图像预测日志。

## 逐段说明

### 条件导入与常量（第 15–40 行）

- 非 pytest；定义支持任务、图名前缀、全局 `_comet_image_prediction_count`。
- 失败则 `comet_ml=None`。

### `_get_comet_mode`（第 43–55 行）

- **若** 设 `COMET_MODE`：deprecated warning 并返回其值；默认 `"online"`。

### `_get_comet_model_name`（第 58–60 行）

- 环境 `COMET_MODEL_NAME` 或默认 `Ultralytics`。

### `_get_eval_batch_logging_interval`（第 63–65 行）

- 环境整数，默认 1。

### `_get_max_image_predictions_to_log`（第 68–70 行）

- 默认 100。

### `_scale_confidence_score`（第 73–76 行）

- 分数乘 `COMET_MAX_CONFIDENCE_SCORE`（默认 100）。

### `_should_log_confusion_matrix`（第 79–81 行）

- `env_bool("COMET_EVAL_LOG_CONFUSION_MATRIX", False)`。

### `_should_log_image_predictions`（第 84–86 行）

- 默认 True。

### `_resume_or_create_experiment`（第 89–121 行）

- **若** `RANK` 非 -1/0：return（DDP 仅主进程）。
- 未设 `COMET_START_ONLINE` 时由 mode 写入 `0/1`。
- `comet_ml.start` + `log_parameters` + `log_others`（评估间隔等）；**except** warning。

### `_fetch_trainer_metadata`（第 124–144 行）

- 计算 `curr_epoch/step`、是否中间 `save_assets`（save_period 且非最终 epoch）。

### `_scale_bounding_box_to_original_image_shape`（第 147–176 行）

- xywhn→xyxy（resize 尺度）→`scale_boxes` 原图→xyxy2xywh→左上角 xywh 列表。

### `_format_ground_truth_annotations_for_detection`（第 179–228 行）

- 筛 `batch_idx==img_idx`；无框 return None。
- 可选类名映射；每框缩放后 append `gt_{label}` 条目。

### `_format_prediction_annotations`（第 231–281 行）

- 用 stem 或整数 image_id 查 `metadata` 预测列表。
- 可选 `class_map` 重映射标签名。
- **若** 有 `faster_coco_eval`：解码 RLE 分割→`cv2.findContours` 多边形点列表。

### `_extract_segmentation_annotation`（第 284–301 行）

- decode 掩码→轮廓；**except** warning 返回 None。

### `_fetch_annotations`（第 304–329 行）

- 合并 GT 与预测非 None 项为单层列表包装。

### `_create_prediction_metadata_map`（第 332–339 行）

- 按 `image_id` 分组 COCO 式预测 dict。

### `_log_confusion_matrix`（第 342–349 行）

- 无矩阵则 return；`log_confusion_matrix` 转置矩阵与类名。

### `_log_images`（第 352–371 行）

- **若有** annotations：zip 路径与标注 `log_image`；**否则**仅图像。

### `_log_image_predictions`（第 374–432 行）

- task 须在 `COMET_SUPPORTED_TASKS`；`jdict` 非空。
- 按 batch 间隔遍历 dataloader；达 `max_image_predictions` 提前 return。
- 每图 `_fetch_annotations` + `_log_images`，计数递增。

### `_log_plots`（第 435–479 行）

- 按 `SegmentMetrics`/`PoseMetrics`/`DetMetrics|OBBMetrics` 拼评估图路径列表。
- 混淆矩阵与（非 classify）labels 图。

### `_log_model`（第 482–485 行）

- `log_model` best.pt。

### `_log_image_batches`（第 488–491 行）

- train/val batch jpg glob。

### `_log_asset` / `_log_table`（第 494–511 行）

- 单文件与 CSV 表。

### `on_pretrain_routine_start`（第 514–518 行）

- 重置图像预测计数；`_resume_or_create_experiment(args)`。

### `on_train_epoch_end`（第 521–531 行）

- 无 running experiment 则 return；`log_metrics` train loss。

### `on_fit_epoch_end`（第 534–575 行）

- metrics、lr；epoch1 记 model info。
- **若** 非 `save_assets`：return。
- 否则 model、可选混淆矩阵、图像预测。

### `on_train_end`（第 578–605 行）

- 模型、plots 条件、混淆矩阵、图像预测、batch 图、results.csv、args.yaml。

### `callbacks`（第 608–618 行）

- 含 teardown walrus 结束 experiment。