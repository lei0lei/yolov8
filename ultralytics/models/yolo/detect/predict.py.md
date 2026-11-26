# `ultralytics/models/yolo/detect/predict.py`

`DetectionPredictor` 继承 `BasePredictor`，在基类预处理与推理不变的前提下，用后处理把原始检测张量经 NMS 映射回原图并封装为 `Results.boxes`。

## 逐段说明

### 类 `DetectionPredictor`（8–31）

属性与方法概览；核心在 `postprocess` 与 `construct_*`。

### `postprocess`（33–81）

- `save_feats`：若存在 `_feats` 则 NMS 需返回保留框索引。
- `nms.non_max_suppression`：使用 `conf`、`iou`（kwargs 可覆盖）、`classes`、`agnostic_nms`、`max_det`；`nc=0`（纯 detect）或类名数量（开放词汇子任务）；`end2end` 来自模型；`rotated` 当 `task==obb`；`return_idxs=save_feats`。
- 非 list 的 `orig_imgs` 转 numpy BGR。
- 若 `save_feats`：`get_obj_feats` 按索引抽特征，NMS 输出取 `[0]`。
- `construct_results` 建结果列表；若有特征则写入各 `Results.feats`。

### `get_obj_feats`（83–100，静态）

- `s` 为各层特征最短通道分组长度。
- 各层 permute、reshape、mean 拼接为 `(B, anchors, s)`。
- 按每张图 NMS 索引切片，无检测时为空张量。

### `construct_results`（102–116）

对每张图调用 `construct_result`，zip `preds`、`orig_imgs`、`self.batch[0]` 路径。

### `construct_result`（118–131）

- `pred[:,:4]` 用 `scale_boxes` 从输入尺寸映射到 `orig_img`。
- 返回 `Results(..., boxes=pred[:,:6])`（xyxy+conf+cls）。