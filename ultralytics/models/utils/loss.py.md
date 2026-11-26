# `ultralytics/models/utils/loss.py`

`DETRLoss` 实现 DETR 主损失与辅助层损失：匈牙利匹配后分类（Focal/VFL/BCE）、L1 框与 GIoU。`RTDETRDetectionLoss` 在标准损失上叠加 CDN 去噪分支，无 `dn_meta` 时将 `*_dn` 置零占位供 DDP。匹配器与 `HungarianMatcher` 成本增益在构造时固定为 class2/bbox5/giou2。

## 逐段说明

### `DETRLoss.__init__(...)`（行 35）

- 默认 `loss_gain` 与 `matcher` 实例化；`FocalLoss`/`VarifocalLoss` 按旗标；`device` 初始 None。

### `_get_loss_class(...)`（行 78）

- one-hot：`scatter_` 到 `nc+1` 再去最后一维；`gt_scores` 与 one-hot 相乘。
- 有 `fl`：有 GT 且 `vfl` 用 VFL，否则 Focal；损失除以 `max(num_gts,1)/nq`。
- 无 `fl`：`BCEWithLogitsLoss` mean+sum 路径。
- 返回 `{loss_class{postfix}: loss * gain["class"]}`。

### `_get_loss_bbox(...)`（行 119）

- GT 空：对 `pred_bboxes[..., :0]` 求和得零损失（防 DDP 未用梯度）。
- 否则 L1 sum/len；GIoU 1-iou 求和/len 乘 `loss_gain["giou"]`。

### `_get_loss_aux(...)`（行 154）

- `use_uni_match` 且无 `match_indices` 时用固定层重匹配。
- 遍历每层 aux 累加 class/bbox/giou 到长度 3 向量，返回 `*_aux{postfix}` 三键。

### `_get_index(match_indices)` 静态（行 201）

拼接 batch 索引、src、dst 张量，返回 `(batch_idx, src_idx), dst_idx`。

### `_get_loss(...)`（行 217）

- 无 `match_indices` 则 matcher；
- 高级索引对齐 pred/gt 框；
- `targets` 填 `nc` 背景类再在匹配位写 `gt_cls`；
- 有 GT 时 `gt_scores` 写 IoU；
- 合并 class+bbox 损失 dict。

### `DETRLoss.forward(...)`（行 260）

- 保存 `device`；对 `pred_bboxes[-1]`/`pred_scores[-1]` 调 `_get_loss`；
- `aux_loss` 时对 `[:-1]` 调 `_get_loss_aux` 并 update。

### `RTDETRDetectionLoss.forward(...)`（行 308）

- **行 328–329**：解包 preds，父类 forward 得 `total_loss`。
- **dn_meta 存在（行 332–341）**：断言 batch 与 `dn_pos_idx` 对齐；`get_dn_match_indices`；对 `dn_bboxes/dn_scores` 再 forward，`postfix="_dn"`，update。
- **否则（行 343–344）**：各现有键对应 `{k}_dn: 0.0` 张量。

### `get_dn_match_indices(...)` 静态（行 348）

每图 `gt_idx.repeat(dn_num_group)`，断言与 `dn_pos_idx[i]` 等长，追加 `(pos, gt)` 元组列表。