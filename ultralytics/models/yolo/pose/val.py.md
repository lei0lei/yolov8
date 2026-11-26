# `ultralytics/models/yolo/pose/val.py`

`PoseValidator` 在检测验证上增加关键点：postprocess 拆 `extra` 为 `keypoints`，`_process_batch` 叠加 OKS tp_p，COCO keypoints JSON 与双指标 eval。

## 逐段说明

### `__init__`（54–70）

`sigma`/`kpt_shape` 待 init；`task=pose`；`PoseMetrics()`。

### `preprocess_targets`（72–76）

`keypoints.float()`。

### `get_desc`（78–92）

Box + Pose 共 10 列指标名。

### `init_metrics`（94–112）

`super`；`kpt_shape` 来自 data；COCO 17×3 用 `OKS_SIGMA` 否则 ones/nkpt 或校验 `kpt_oks_sigmas` 长度与正值。

### `postprocess`（114–139）

父类 NMS 后 `extra` reshape 为 `(-1,*kpt_shape)` 改名 `keypoints`。

### `_prepare_batch`（141–162）

`super` 后对当前图 keypoints：x*=w，y*=h（letterbox 像素空间）。

### `_process_batch`（164–191）

`tp` 来自父类 box IoU；无 GT/pred 时空 `tp_p`；否则面积 `xywh prod * 0.53`，`kpt_iou` + `match_predictions` 得 `tp_p` 并入 dict。

### `gather_stats`（193–196）

`super` + `_gather_image_metrics(self.metrics.pose)`。

### `save_one_txt`（198–219）

`Results` 含 boxes 与 keypoints。

### `pred_to_json`（221–240）

`super` 写 bbox 项后给最近 len(kpts) 条 jdict 追加 `keypoints` 展平列表。

### `scale_preds`（242–252）

父类 scale boxes + `scale_coords` keypoints 到原图。

### `eval_json`（254–258）

`person_keypoints_val2017.json`；`coco_evaluate` bbox+keypoints，后缀 Box/Pose。