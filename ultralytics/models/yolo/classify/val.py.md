# `ultralytics/models/yolo/classify/val.py`

`ClassificationValidator` 扩展 `BaseValidator`：累积 top-k 预测与 GT，用 `ClassifyMetrics` 与混淆矩阵；相对检测验证无 NMS/IoU。

## 逐段说明

### `__init__`（56–69）

`super`；`targets`/`pred` 置 None；`task=classify`；`metrics=ClassifyMetrics()`。

### `get_desc`（71–73）

表头 classes / top1_acc / top5_acc。

### `init_metrics`（75–81）

`names`、`nc`；清空 pred/targets；`ConfusionMatrix(task=classify)`。

### `preprocess`（83–87）

`preprocess_targets` 后 img 上 device；quantize==16 则 half 否则 float。

### `update_metrics`（89–102）

`n5 = min(len(names), 5)`；每 batch 对 logits `argsort` 降序取前 n5 类写入 `pred`；`targets` 存 `batch["cls"]` int32 CPU。

### `finalize_metrics`（104–125）

`confusion_matrix.process_cls_preds`；plots 时归一化/非归一化各画一张；写 metrics.speed/save_dir/confusion_matrix。

### `postprocess`（127–129）

list/tuple 输出取 `[0]`。

### `get_stats`（131–134）

`metrics.process(targets, pred)` → `results_dict`。

### `gather_stats`（136–147）

rank0 `gather_object` 合并 pred/targets 列表；其它 rank 发送空。

### `build_dataset`（149–151）

`ClassificationDataset` 无增强。

### `get_dataloader`（153–166）

无图抛错；`build_dataloader` rank=-1。

### `print_results`（168–171）

一行 `all` + top1/top5。

### `plot_val_samples`（173–191）

`batch_idx` + `plot_images` 保存 labels 图。

### `plot_predictions`（193–218）

构造 `cls=argmax`、`conf=amax`，保存 pred 图。