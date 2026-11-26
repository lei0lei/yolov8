# `ultralytics/models/yolo/segment/val.py`

`SegmentationValidator`：检测 NMS 后解码 mask，`mask_iou` 得 `tp_m`，RLE 写 COCO segm，双通道 mAP（Box/Mask）。

## 逐段说明

### `__init__`（36–48）

`task=segment`；`SegmentMetrics()`；`process` 函数指针稍后设。

### `preprocess_targets`（50–61）

`masks.float()`。

### `init_metrics`（63–75）

`super`；save_json 时检查 faster-coco-eval；`process` 选 native（json/txt/tile）或 letterbox mask。

### `get_desc`（77–91）

Box + Mask 十列表头。

### `postprocess`（93–108）

取 `proto`；父类 NMS on `preds[0]`；imgsz=proto 空间×4；每图 `process(proto[i], extra coeffs, bboxes, shape)` → `masks`。

### `_prepare_batch`（110–135）

`super` 后处理 GT masks：`overlap_mask` 则按实例索引 one-hot；否则按 batch_idx 切片；必要时 interpolate 到 mask_size；二值 `gt_(0.5)`。

### `gather_stats`（137–140）

`super` + seg 的 image metrics。

### `_process_batch`（142–173）

父类 box tp；mask 分辨率不一致则 stride 下采样对齐；`mask_iou` flatten → `tp_m`。

### `plot_predictions`（175–192）

mask 截断 `max_det` 转 uint8 CPU 再调父类绘图。

### `save_one_txt`（194–211）

`Results` 含 uint8 masks。

### `pred_to_json`（213–286）

内嵌 `to_string`（RLE delta+5bit 编码）、`multi_encode`（GPU transition 打包）；transpose 预测 mask；`super` 后附 `segmentation` RLE dict。

### `scale_preds`（288–297）

`scale_masks`；tile 分支 gt 0.5 与子采样 4 对齐；否则 byte。

### `eval_json`（299–307）

`coco_evaluate` bbox+segm → Box/Mask 后缀。