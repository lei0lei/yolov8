# `ultralytics/models/yolo/obb/val.py`

`OBBValidator` 继承 `DetectionValidator`：`task=obb`、`OBBMetrics`、probiou 匹配、框 tensor 含角度维；DOTA 导出与瓦片合并 NMS。

## 逐段说明

### `__init__`（47–61）

`task=obb`；`metrics=OBBMetrics()`。

### `init_metrics`（63–72）

`super` 后判 `is_dota`（路径含 DOTA）；`confusion_matrix.task=obb`。

### `_process_batch`（74–96）

无目标空 tp；否则 `batch_probiou` + `match_predictions`。

### `postprocess`（98–110）

父类 NMS 后每 pred 把 `extra`（角度）拼到 `bboxes` 末维。

### `_prepare_batch`（112–144）

有目标时 `bbox[..., :4]` 乘 imgsz 宽高（xywhr 前四维）；不转 xyxy（与检测版不同）。

### `plot_predictions`（146–178）

批预测不转 xywh；直接 `plot_images` 画旋转框。

### `pred_to_json`（180–208）

`rbox` 列表 + `poly` 八点；category_id 经 `class_map`。

### `save_one_txt`（210–238）

用零图占位构造 `Results`，`obb` 为 `cat([bboxes, conf.unsqueeze(-1), cls.unsqueeze(-1)])`，调用 `save_txt` 写 DOTA/YOLO-OBB 归一化格式。

### `scale_preds`（240–247）

`scale_boxes(..., xywh=True)` 含角度。

### `eval_json`（249–317）

DOTA 且 save_json：删旧 Task1_*.txt；从 predictions.json 按类写 `predictions_txt/Task1_{class}.txt` 多边形行。
- **merged**：解析 `___` 瓦片偏移，按图合并 rbox，用 `max_wh` 类偏移 + `TorchNMS.fast_nms(..., batch_probiou, 0.3)`；写 `predictions_merged_txt`。
- 返回 stats（不跑 COCO eval，与检测 eval_json 分支不同）。