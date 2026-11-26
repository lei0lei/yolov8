# `ultralytics/utils/nms.py`

本模块在推理与验证的后处理阶段对模型原始输出做非极大值抑制（NMS），去掉重叠且置信度较低的冗余框。标准检测头输出为 `(batch, 4+nc+extra, num_anchors)`，先按置信度筛候选，再把 `xywh` 转为 `xyxy`，可选多标签或仅取最大类，最后用 IoU 阈值合并；旋转框（OBB）用 ProbIoU，端到端（end2end）模型则只做置信度与类别过滤。验证器与 `predictor` 在得到 logits 后调用 `non_max_suppression`；`nms_boxes` 在已按类别平移坐标后选择 torchvision、原生 XPU 或自研 `TorchNMS` 实现。训练阶段不经过 NMS，但 TAL 与损失里用的 IoU/CIoU 与这里评测用的几何量同源（`metrics` 模块）。

## 逐段说明

### `non_max_suppression`（第 13～157 行）——参数校验与 end2end 分支

断言 `conf_thres` 与 `iou_thres` 落在 [0,1]。若 `prediction` 为元组（验证时含 loss 输出），只取推理张量。`classes` 转为同设备张量。若最后一维为 6 或 `end2end=True`，逐图用置信度与可选类别掩码筛索引，截断到 `max_det`，直接返回 `(x1,y1,x2,y2,conf,cls[, extra...])` 列表，不做 IoU NMS。

### `non_max_suppression`（第 81～157 行）——标准 BCN 路径主循环

将 `(B, C, N)` 转为 `(B, N, C)`。`nc` 默认从通道数推断；`mi=4+nc` 为 mask/extra 起始下标。先用各类别最大 logit 是否超过 `conf_thres` 得每图候选掩码 `xc`。对每张图：筛候选；非旋转时前四通道 `xywh2xyxy`；按 `multi_label` 展开为多类检测或取 `cls.max` 单类；再按 `classes` 过滤；若框数超过 `max_nms` 按置信度取 top；为类感知 NMS，把 `class_id * max_wh`（agnostic 时为 0）加到框坐标上使不同类在坐标空间分离；旋转框拼接 `(x,y,w,h,angle)` 五元组再调 NMS；结果截断 `max_det`；超时则告警并跳出。`return_idxs=True` 时另返回原始 anchor 索引。

### `nms_boxes`（第 160～169 行）

旋转：`TorchNMS.fast_nms`，IoU 函数为 `batch_probiou`。水平框：非 XPU 且已导入 torchvision 时用 `torchvision.ops.nms`；否则 `TorchNMS.nms`。输入框应已含类别偏移。

### 类 `TorchNMS`（第 172～187 行）

提供与 torchvision 行为一致的常规 NMS 及基于上三角 IoU 矩阵的 Fast-NMS，供 YOLO 导出与 OBB 使用。

### `TorchNMS.fast_nms`（第 189～241 行）

按分数降序排列框。`iou_func` 计算两两 IoU（OBB 时为 pairwise ProbIoU）。`use_triu=True`：取上三角，`triu(diagonal=1)` 后，每列和为 0 的列索引为保留框（不被更高分框压制）。`use_triu=False`：手工上三角掩码，将被抑制的分数置零，用 `topk` 固定长度索引以适配 TFLite 导出。返回原排序下的保留索引。

### `TorchNMS.nms`（第 243～301 行）

空输入返回空 long 索引。一次性解包 `x1,y1,x2,y2` 与面积，按分数排序。贪心：取当前最高分框，与其余框算交并比，保留 IoU ≤ 阈值的框进入下一轮；若无交集则跳过 IoU 计算直接保留全部剩余。与 torchvision 早停逻辑对齐。