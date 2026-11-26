# `ultralytics/utils/tal.py`

任务对齐分配器（Task-Aligned Assigner，TAL）在训练前向的每次 loss 计算里，把 GT 框分配到多尺度 anchor 点，并产生软分类目标与回归目标。检测损失（`v8DetectionLoss` 等）先用 `make_anchors` 生成格点，解码预测框后把 **sigmoid 分类分** 与 **像素空间预测框** 送入 assigner；返回的 `target_scores` 经对齐度量归一化后作为 BCE 标签，正样本 `fg_mask` 上的框 loss 再乘 `target_scores` 各类求和权重。验证阶段不用 TAL，但 anchor 编解码函数 `dist2bbox` / `bbox2dist` 与训练解码一致。姿态、分割、旋转框分别通过子类把重叠度从 CIoU 换成 OKS、框+掩码 IoU 或 ProbIoU。

## 逐段说明

### 类 `TaskAlignedAssigner`（第 14～61 行）`__init__`

超参：`topk`（每 GT 保留的 anchor 候选数，默认 13）、`topk2`（冲突后再筛，默认同 topk）、`num_classes`、`alpha`/`beta`（对齐度量中分类与定位的幂次，常见 0.5 与 6）、各层 `stride`、极小边 `stride_val`（细边 GT 在候选池内放大）、`eps` 防除零。

### `TaskAlignedAssigner.forward`（第 63～140 行）

`@torch.no_grad()`。无 GT 时返回背景类标签、`num_classes` 占位、全零 target 与空前景。正常调用 `_forward`；CUDA OOM 时按图逐 batch 切片重试并拼回，不改变训练 batch size。

### `TaskAlignedAssigner._forward`（第 142～181 行）

调用 `get_pos_mask` 得 `mask_pos`、`align_metric`、`overlaps`；`select_highest_overlaps` 解决一 anchor 多 GT，得 `target_gt_idx` 与 `fg_mask`；`get_targets` _gather GT 类别与 xyxy 框，one-hot 乘前景掩码。对齐分数归一化：在正样本掩码内按 GT 取最大对齐度与最大 IoU，逐 GT 用 `align * pos_overlaps / (pos_align + eps)` 缩放，再对 anchor 维取 max 乘到 `target_scores` 上，实现软标签质量加权。

### `get_pos_mask`（第 183～210 行）

`select_candidates_in_gts`：anchor 中心落在 GT 内（过小 GT 边长抬到 `stride_val`）。`get_box_metrics` 在「中心在框内且有效 GT」处算对齐度与 IoU。`select_topk_candidates` 按 `align_metric` 每 GT 取 topk。三者与 `mask_gt` 相与得最终正样本候选掩码。

### `get_box_metrics`（第 212～239 行）

在 `mask_gt` 为真的 (batch, gt, anchor) 三元组上：取该 GT 类对应的 `pd_scores`；`pair_overlaps` 得 IoU；**对齐度量 = score^α × IoU^β**。用 scatter 填回 `(b, n_max_obj, n_anchors)` 稠密张量。

### `pair_overlaps` / `iou_calculation`（第 241～255 行）

水平框：`bbox_iou(..., CIoU=True)` 后 `clamp(0)`。

### `select_topk_candidates`（第 257～284 行）

沿 anchor 维 `topk` 最大对齐度；可选 `topk_mask` 屏蔽无效 GT。同一 anchor 被同一 GT 的 topk 列表重复选中时计数 >1 则清零（避免重复正样本）。

### `get_targets`（第 286～324 行）

用 `target_gt_idx` 展平索引取 GT 类别与框；类别 clamp 非负；scatter one-hot 得 `target_scores`，再乘 `fg_mask`。

### `select_candidates_in_gts`（第 326～357 行）

GT xyxy→xywh，过小宽高替换为 `stride_val` 再转回 xyxy。anchor 中心相对 GT 左上/右下均大于 `eps` 则在框内。

### `select_highest_overlaps`（第 359～391 行）

对 anchor 维求和得前景计数；若 >1 表示多 GT 竞争，仅保留 IoU 最大 GT 的正样本。若 `topk2 != topk`，再按对齐度 topk2 收紧 `mask_pos`。沿 GT 维 argmax 得每 anchor 的 `target_gt_idx` 与是否为前景。

### 类 `PoseTaskAlignedAssigner`（第 394～437 行）

`level_range=(2,12)`：GT 等效边长 `sqrt(area)` 需落在 `[lo*s, hi*s]` 与 anchor stride `s` 匹配，最粗/最细层放宽；若无候选则退回未门控掩码。`pair_overlaps`：可见关键点时用 COCO 式 OKS（与验证一致，面积 ×0.53），否则回退框 IoU。`forward` 需 `anc_strides`。

### 类 `SegTaskAlignedAssigner`（第 440～494 行）

`get_box_metrics` 向 `pair_overlaps` 传入 scores/labels。`pair_overlaps`：默认框 IoU；对每个 GT 仅对 **3×topk** 个框度量最高的 anchor 解码 mask（系数×原型，阈值化并 crop 到预测框），与 GT mask 算 IoU，未选中者掩码 IoU 为 0；**最终重叠 = (IoU + mask IoU) / 2**。

### 类 `RotatedTaskAlignedAssigner`（第 497～537 行）

`iou_calculation` 用 `probiou`。`select_candidates_in_gts`：过小 wh 抬升后，用平行四边形投影判断 anchor 中心是否在旋转框内。

### `make_anchors`（第 540～566 行）

各 FPN 层按 `stride` 与 `grid_cell_offset`（0.5 为格心）堆叠 `(sx,sy)`，拼接 `stride_tensor` `(N,1)`。

### `dist2bbox`（第 569～588 行）

距离 `(l,t,r,b)` 相对 anchor：中心±半宽得到 xyxy，或合并为中心与 wh。

### `bbox2dist`（第 591～606 行）

xyxy 相对 anchor 的 ltrb；可选 `reg_max` 上界钳位，供 DFL 目标。

### `dist2rbox` / `rbox2dist`（第 609～666 行）

旋转框：用预测角将 l/r 偏移旋到图像坐标，输出 `xywh`（不含角）；逆变换把 GT 旋转框变为 anchor 系 ltrb，可选 `reg_max` 钳位。