# `ultralytics/utils/loss.py`

本文件集中定义 YOLO 家族各任务在 `engine/trainer` 前向里调用的 **损失准则**：检测类通过 `TaskAlignedAssigner`（`tal.py`）产生软分类目标与正样本掩码，再组合分类 BCE、CIoU/ProbIoU 框损与 DFL（或 L1）回归；超参 `hyp.box/cls/dfl/pose/...` 在返回前乘到各项上，总损失再乘 batch size 供反向。分割叠加原型系数 BCE-crop 掩码损与可选语义分支；姿态加 OKS 式关键点损与可见性 BCE，YOLO26 姿态可再接 RLE 流模型；OBB 换 ProbIoU 与角度 sin² 损；深度、语义分割、E2E 双分支与 TVP 等有独立入口。验证指标在 `metrics.py` 中计算，但训练框 IoU 与 `bbox_iou`/`probiou` 与此处一致。

## 逐段说明

### 类 `VarifocalLoss`（第 30～59 行）

`forward(pred_score, gt_score, label)`：权重 `α·σ(pred)^γ·(1-label) + gt_score·label`；对 logits 做 float 精度 BCE，按 anchor 维 mean 再 sum。用于可选的分类替代（OBB 注释中有 VFL 写法）。

### 类 `FocalLoss`（第 62～94 行）

logits BCE 乘 `(1-p_t)^γ`，`p_t` 为正类概率或负类补集；可选 `α` 平衡因子。按 anchor mean 后 sum。

### 类 `DFLoss`（第 97～116 行）

分布焦点损失：`target` 夹在 `[0, reg_max-1-ε)`，左右整数 bin 为 `tl/tr`，线性权重 `wl/wr`；对 `pred_dist` softmax 后 gather 两 bin 的负 log 概率加权平均，输出 per-anchor 标量。

### 类 `BboxLoss`（第 119～164 行）

`reg_max>1` 时挂 `DFLoss`。`forward`：用 `fg_mask` 索引正样本；**weight = target_scores 在正样本上各类之和**；**loss_iou = Σ(1-CIoU)·weight / target_scores_sum**（预测与 GT 均为 xyxy）；DFL 目标 `bbox2dist(anchor, target_bboxes, reg_max-1)`；**loss_dfl = DFL(pred)·weight 求和 / target_scores_sum**。无 DFL 时 ltrb 按 stride 与 `imgsz` 归一化后用 L1。

### 类 `RLELoss`（第 167～227 行）

残差对数似然：`log_σ - log_φ + (residual 时) log(2σ)+|error|`，可乘 `target_weight`；`size_average` 时除以元素数再 sum。供姿态流模型不确定性。

### 类 `RotatedBboxLoss`（第 230～276 行）

继承 `BboxLoss`，IoU 换 **ProbIoU**（`floor=0.01`）；DFL 目标用 `rbox2dist` 含角度；无 DFL 路径同样归一化 L1。

### 类 `MultiChannelDiceLoss`（第 279～309 行）

sigmoid 后与 target 算每通道 Dice，`1-Dice` 再对通道 mean/sum/none。

### 类 `BCEDiceLoss`（第 312～333 行）

`0.5·BCE + 0.5·Dice`；target 空间尺寸与 pred 不一致时 nearest 下采样。

### 类 `KeypointLoss`（第 336～352 行）

欧氏距离平方 `d`；**e = d / (2σ)² / (area·2)**（COCO eval）；损失 `(1-exp(-e))·kpt_mask`，再乘 **kpt_loss_factor = K / 可见点数** 的 batch 归一。

### 类 `v8DetectionLoss`（第 355～512 行）

**`__init__`（358～391）**：读 Detect 头 `stride/nc/reg_max`；`TaskAlignedAssigner(topk, α=0.5, β=6)`；`BboxLoss`；DFL 用 `proj=0..reg_max-1`；可选 `class_weights`。

**`preprocess`（393～412）**：batch 内 GT 装箱为 `(B, max_n, 5)`，xywh→xyxy 乘 `scale_tensor`。

**`bbox_decode`（414～421）**：有 DFL 时对每边 softmax×proj 得期望距离，再 `dist2bbox`→xyxy。

**`get_assigned_targets_and_loss`（423～490）**：permute 框与分类；`make_anchors`；拼 batch GT；assigner 输入为 **detach 的 sigmoid 分与像素预测框**、像素 anchor、归一化 GT；**BCE(pred_scores, target_scores)**，可类权重；`bbox_loss` 用格点空间 target 框；**乘 hyp.box/cls/dfl**。

**`parse_output` / `__call__` / `loss`（492～512）**：取训练 dict；`loss` 再乘 batch_size。

### 类 `v8SegmentationLoss`（第 515～677 行）

在检测三项外加 **seg_loss、sem_loss**；`BCEDiceLoss` 用于语义头。

**`loss`（527～587）**：先 `get_assigned_targets_and_loss`；前景存在时 `calculate_segmentation_loss`；`pred_semantic` 时对未覆盖像素 scatter 的多类 one-hot 算 BCE+Dice（增益误用 `hyp.box` 注释为 seg）；无前景时对 proto/mask 零乘保持 DDP 图连通。

**`single_mask_loss`（589～612）**：`einsum` 系数×原型得 logits mask，BCE 后 **crop 到 xyxy**，每实例 **/area（归一化框面积）** 再 sum。

**`calculate_segmentation_loss`（614～677）**：逐图取前景 anchor 索引，overlap 模式从单通道实例 id 掩码切 GT，否则按 batch 索引；累加后 **/fg_mask.sum()**。

### 类 `SegmentLoss26`（第 680～708 行）

替换为 `SegTaskAlignedAssigner`；`get_assigned_targets_and_loss` 构造 `gt_masks`、`mask_scale`、`detach` 的系数与 proto 传入 assigner，使正样本选择考虑掩码 IoU。

### 类 `v8PoseLoss`（第 711～839 行）

损失项：box、pose、kobj、cls、dfl。`KeypointLoss` 用 OKS_SIGMA 或自定义 sigmas。

**`loss`（737～772）**：检测分配后 `kpts_decode`（×2 偏移 anchor-0.5）；`calculate_keypoints_loss`；**hyp.pose / hyp.kobj**。

**`kpts_decode`（774～781）**：格点系关键点坐标。

**`_batch_keypoints`（783～794）**：按图 pad 关键点并乘 `imgsz` 变像素坐标。

**`calculate_keypoints_loss`（796～839）**：正样本上 GT 关键点除 stride；面积来自 xywh 积；可见性 mask 上 KeypointLoss；三维输出时对第 3 通道 BCE 可见性。

### 类 `PoseLoss26`（第 842～1018 行）

**`__init__（845～861）**：可选 `RLELoss`+`flow_model`；`PoseTaskAlignedAssigner`；RLE 关节权重 `RLE_WEIGHT`。

**`loss`（863～914）**：YOLO26 解码为 anchor 直接加偏移；分配前传入 detach 像素关键点；RLE 项 **hyp.rle**。

**`calculate_rle_loss`（928～968）**：可见点标准化误差进 flow `log_prob`；NaN/Inf 过滤；`RLELoss(σ, log_φ, error, weights)`。

**`calculate_keypoints_loss`（970～1018）**：三返回值含 RLE。

### 类 `v8ClassificationLoss`（第 1021～1033 行）

`cross_entropy(preds, batch["cls"], class_weights)`，无 TAL。

### 类 `v8OBBLoss`（第 1036～1207 行）

**`__init__`（1039～1051）**：`RotatedTaskAlignedAssigner`、`RotatedBboxLoss`；多 **angle_loss**。

**`preprocess`（1053～1072）**：GT 六元（cls+xywhr）按图装箱并缩放。

**`loss`（1074～1158）**：过滤过小旋转框；`bbox_decode`→`dist2rbox`+angle；分配用像素 xywhr；BCE 与旋转 bbox_loss；**angle_loss = scale_weight·sin²(2Δθ)·weight**，`scale_weight=exp(-(log(w/h))²/λ²)`，Δθ 包裹到 [-π/2,π/2)；**hyp.angle**。

**`bbox_decode`（1160～1176）**：DFL+`dist2rbox` 拼角度。

**`calculate_angle_loss`（1178～1207）**：仅前景，除 `target_scores_sum`。

### 类 `DepthLoss26`（第 1210～1301 行）

**`__init__`（1217～1226）**：`silog_weight/grad_weight/silog_lambda`；多尺度梯度 4 层。

**`_grad_l1`（1228～1239）**：有效 mask 双边门控的 log 深度 dx/dy L1。

**`__call__`（1241～1301）**：对齐尺寸；有效像素 <10 返回零图梯度；**SILog**：log 差去均值后方差+ `(1-λ)m²` 开方；**grad_loss** 原分辨率加池化下采样层（ occupied>0.7 的图继续）。返回 **×batch** 与分项 dict。

### 类 `E2EDetectLoss`（第 1304～1321 行）

`one2many` topk=10 + `one2one` topk=1 两路 `v8DetectionLoss`，总损失与 loss dict 逐项相加。

### 类 `E2ELoss`（第 1324～1356 行）

可插拔 `loss_fn`；one2one 用 `topk=7, topk2=1`；训练初 **o2m=0.8** 线性衰减到 **0.1**（`decay` 按 epoch）；返回加权总和，**loss_items 仅 one2one 分支**。

### 类 `TVPDetectLoss`（第 1359～1381 行）

包装 `v8DetectionLoss`，日志名去掉 `_loss` 后缀；仅返回总损失第二分量（设计为 VP 分类相关）。

### 类 `TVPSegmentLoss`（第 1384～1398 行）

包装 `v8SegmentationLoss`；**只返回 cls_loss 标量**（索引 2），不含 sem_loss。

### 类 `SemanticSegmentationLoss`（第 1401～1513 行）

**`__init__`（1410～1433）**：Semantic 头 `nc`；Cityscapes 可用 `CITYSCAPES_WEIGHT`；单类 BCE，多类 CE ignore 255。

**`_ce_loss`（1443～1453）**：逐像素 CE/BCE sum，分母为有效像素或加权类和。

**`_dice_loss` / `_binary_dice_loss`（1455～1476）**：多类 softmax×one-hot 或二值 sigmoid Dice。

**`forward`（1478～1513）**：主分支 CE+Dice；aux 头 CE×0.4；返回 **total×B** 与 ce/dice/aux 分项。