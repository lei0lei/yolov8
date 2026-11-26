# `ultralytics/models/yolo/semantic/val.py`

`SemanticSegmentationValidator` 继承 `DetectionValidator` 骨架但重写指标路径：逐像素 `SemanticMetrics`、无检测 NMS，`postprocess` 为 argmax 类图，可选 PNG 导出预测。

## 逐段说明

### `__init__`（38–52）

`task=semantic`；`SemanticMetrics()`；`_semantic_target_shape` 缓存。

### `init_metrics`（54–75）

`names/nc`；`SemanticMetrics(names)`；`seen=0`；save_json 时 `results_dir`、`mask_root`；二值单类混淆矩阵名扩 background；`ConfusionMatrix(task=semantic)`。

### `preprocess_targets`（77–89）

`semantic_mask` int32 上 device；记 target 空间 shape。

### `postprocess`（91–112）

tuple 取 [0]；3D 类图 nearest 对齐 target 后 int32；4D logits 双线性对齐再 argmax/gt(0)。

### `update_metrics`（114–124）

`save_json` 时 `save_pred_masks`；`update_stats(preds, semantic_mask)`；`seen+=B`。

### `gather_stats`（126–140）

初始化空 matrix 若需；`dist.reduce` matrix SUM；rank0 gather `nt_per_image` 求和。

### `save_pred_masks`（142–161）

scale_masks 到原图；可选 `label_mapping` 反变换 ID；相对 `mask_root` 写 PNG 树。

### `get_stats`（163–173）

`metrics.process`；混淆矩阵转置写入 `confusion_matrix.matrix`。

### `get_desc`（175–181）

Class/Images/Pixels/mIoU/PixAcc。

### `print_results`（183–187）

`super` 后提示 masks 保存目录。

### `plot_predictions`（189–199）

`plot_images` 键 `semantic_mask`。