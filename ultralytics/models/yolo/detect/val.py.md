# `ultralytics/models/yolo/detect/val.py`

`DetectionValidator` 实现标准 YOLO 检测验证：NMS 后逐图匹配 IoU、`DetMetrics`、COCO/LVIS JSON、混淆矩阵与 DDP 聚合；是 obb/pose/segment/semantic/depth/yoloe 等子类的基础。

## 逐段说明

### `__init__`（45–61）

初始化 COCO/LVIS 标志、`class_map`、`task=detect`、`iouv` 0.5:0.05:0.95（10 点）、`DetMetrics()`。

### `_check_max_det`（63–95，静态）

各 split 统计标签最大实例数；超 `args.max_det` 警告；默认 cfg 且用户未自定义时抬高 `max_det` 到观测值。

### `preprocess`（97–109）

`preprocess_targets`；img → device；half/float 由 quantize；除 255。

### `init_metrics`（111–158）

`save_txt` 时清 labels 目录；`label_root`；非训练时 `_check_max_det`；判 `is_coco`/`is_lvis`；`class_map` COCO91 或 1..nc；`save_json` 条件；end2end 时 head `max_det`≥300；`jdict`/`gdict`/`eval_ids`/`pred_counts`；自定义 JSON+tile 时 eval_ids 去重 source_index；`confusion_matrix`；metrics 初始化与 `area_ap`。

### `get_desc`（160–162）

检测 mAP 表头。

### `postprocess`（164–186）

NMS：`single_cls` 且 nc==1 时 classes 过滤为 None；`multi_label=True`；返回每图 dict bboxes/conf/cls/extra。

### `_prepare_batch`（188–213）

按 `batch_idx==si` 取 cls/bbox；有目标则 xywh→xyxy 乘 letterbox 尺寸。

### `_prepare_pred`（215–226）

`single_cls` 时 pred cls 乘 0。

### `update_metrics`（228–293）

逐图 `seen+=1`；`_prepare_batch`；自定义 gdict 写 GT COCO 框；`_prepare_pred` + `metrics.update_stats`（含 tp、target_cls、conf 等）；plots 更新混淆矩阵与 visualize；无预测跳过保存；否则 scale_preds 后 json/txt。

### `finalize_metrics`（295–302）

画混淆矩阵；写 speed/cm/save_dir。

### `_gather_image_metrics`（304–315）

DDP gather 单指标对象的 `image_metrics`。

### `gather_stats`（317–348）

rank0 合并 stats、jdict、gdict、pred_counts；`_gather_image_metrics(box)`；混淆矩阵 sum reduce。

### `get_stats`（350–363）

`metrics.process`；detect+save_json 时占位 small/medium/large mAP 键；训练中 `eval_json`；`clear_stats`。

### `print_results`（365–383）

汇总行；无标签 warning；verbose 非训练按类打印。

### `_process_batch`（385–399）

无 GT 或无 pred 返回空 tp；否则 `box_iou` + `match_predictions`。

### `build_dataset`（401–415）

`get_split_fraction` + `build_yolo_dataset`。

### `get_dataloader`（417–440）

模型类数少于 data nc 时裁剪 labels；val dataloader 不 shuffle。

### `plot_val_samples` / `plot_predictions`（442–485）

标准标签/预测拼图；预测转 xywh 批量化。

### `save_one_txt`（487–503）

用 `Results.save_txt` 写归一化检测行。

### `pred_to_json`（505–536）

image_id 来自 stem；xyxy→xywh 左上角；追加 jdict。

### `scale_preds`（538–548）

`scale_boxes` 到原图。

### `eval_json`（550–573）

gdict 或 jdict 路径；选 COCO/LVIS 标注文件；`coco_evaluate`。

### `coco_evaluate`（575–642）

需 jdict 且（自定义 gdict 或 COCO/LVIS 且非 single_cls）；faster-coco-eval；多 iou_types；gdict 时 imgIds 来自 gdict images；可选 catIds 过滤；训练外写 mAP50/mAP50-95/fitness；area AP；LVIS APr/c/f；异常 warning。