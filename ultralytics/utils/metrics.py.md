# `ultralytics/utils/metrics.py`

本文件为 **验证（val）与超参搜索** 提供几何相似度、PR 曲线、混淆矩阵与各任务 `*Metrics` 聚合器；检测/分割/姿态/OBB 验证器在每图匹配后向 `DetMetrics` 等追加 `tp/conf/pred_cls/target_cls`，epoch 末 `process()` 调 `ap_per_class` 得 mAP。训练时 `loss.py` 与 `tal.py` 直接调用本文件的 `bbox_iou`、`probiou`、`OKS_SIGMA` 等，与评测 IoU 定义对齐。预测后处理 `nms.py` 用 `box_iou`/`batch_probiou`。常量 `RLE_WEIGHT`、`CITYSCAPES_WEIGHT` 亦被损失模块引用。

## 逐段说明

### 模块常量（第 19～51 行）

`OKS_SIGMA`：COCO 17 点 OKS 尺度；`RLE_WEIGHT`：姿态 RLE 关节权重；`CITYSCAPES_WEIGHT`：语义分割 19 类频率权重。

### `bbox_ioa`（第 54～83 行）

NumPy：N×M 轴对齐框，默认 **交 / box2 面积**；`iou=True` 时为标准 IoU。

### `box_iou`（第 86～106 行）

Torch N×M xyxy，pairwise 交并比，float 精度。

### `bbox_iou`（第 109～175 行）

末维 4 可广播；`xywh` 时先转 xyxy。基础 IoU = inter/union；**GIoU** 减凸包过剩面积比；**DIoU** 再减中心距离/凸对角²；**CIoU** 再加宽高比一致项 `v·α`（α 停梯度由 `v` 与 `1-iou` 比决定）。训练框损与 TAL 用 **CIoU**。

### `mask_iou`（第 178～196 行）

扁平化 mask 矩阵乘得交集，并集 = 面积和 − 交，分母加 eps。

### `kpt_iou`（第 199～218 行）

OKS：`exp(-d/(2σ)²/(area·2))` 对可见关键点求平均，得 N×M 相似度（验证匹配）。

### `_get_covariance_matrix`（第 222～240 行）

xywhr 高斯框：`w²/12、h²/12` 与角得协方差参数 a,b,c。

### `probiou`（第 243～288 行）

成对旋转框 ProbIoU：中心差与协方差合成为 `bd`，`iou = 1 - sqrt(1-exp(-bd))`；可选 CIoU 宽高比罚项。OBB 训练与验证框相似度。

### `batch_probiou`（第 291～324 行）

N×M 全对 ProbIoU，NMS OBB 与混淆矩阵 OBB 匹配用。

### `smooth_bce`（第 327～340 行）

标签平滑正/负 BCE 目标 `(1-0.5ε, 0.5ε)`。

### 类 `ConfusionMatrix`（第 343～667 行）

**`__init__`（355～372）**：任务类型决定矩阵是否含 background 行列；可选 `save_matches`。

**`_append_matches`（374～397）**：可视化用 TP/FP/FN/GT 批数据追加。

**`process_cls_preds`（399～409）**：分类任务，预测 top-1 对角计次。

**`process_batch`（411～488）**：检测/OBB：置信度过滤后 **类无关** greedy 匹配（与 val 同 IoU 阈值，OBB 用 batch_probiou）；正确类为 TP 对角，错类同时 FP+FN；未匹配检测为 FP 列，未匹配 GT 为 FN 行。

**`tp_fp`（490～500）**：对角为 TP，行和减 TP 为 FP。

**`plot_matches`（502～543）**：四宫格 GT/FP/TP/FN 图。

**`plot`（545～618）**：归一化热图、大类下采样、保存 PNG。

**`print` / `summary`（620～667）**：控制台与导出字典。

### `smooth`（第 670～675 行）

一维 box 滤波平滑曲线。

### `plot_pr_curve` / `plot_mc_curve`（第 679～768 行）

PR 与 metric-confidence 曲线绘图回调。

### `compute_ap`（第 775～796 行）

单调 precision 包络 + 101 点插值（COCO）求 AP。

### `ap_per_class`（第 799～894 行）

按 conf 降序；每类累积 TP/FP 曲线；recall=TP/labels，precision=TP/(TP+FP)；多 IoU 列各算 AP；可选画 PR/F1/P/R 曲线；**F1 均值平滑后 argmax** 得报告点 p/r/f1 与 tp/fp 估计。返回 12 元组供 `Metric.update`。

### 类 `Metric`（第 897～1100 行）

存 per-class p/r/f1、`all_ap (nc,10)`、曲线数据。属性 **map50/map75/map**、**fitness**=0.25 权重仅 **mAP50-95**。`update_image_metrics`：单图 IoU0.5 TP 计 P/R/F1；空图空预测记 1.0。

### 类 `DetMetrics`（第 1103～1299 行）

**`__init__`（1132～1144）**：`box=Metric`，stats 列表，速度字典。

**`update_stats`（1146～1155）**：追加 tp/conf/... 并更新单图 box 指标。

**`process`（1157～1186）**：concat 后 `ap_per_class`→`box.update`；`nt_per_class`/`nt_per_image` 直方图。

**`summary`（1242～1279）**：每类 Images/Instances/P/R/F1/mAP50/mAP75/mAP50-95。

**`_task_summary`（1281～1299）**：子任务 Metric 列前缀扩展（Mask/Pose）。

其余 property：`keys`、`results_dict`、`curves`、委托 `box` 的 mean/maps/fitness。

### 类 `SegmentMetrics`（第 1302～1444 行）

增加 `seg` Metric 与 **`tp_m`**；`process` 在 box 后再对 mask TP 跑 `ap_per_class(prefix="Mask")`；`fitness` 为 seg+box fitness 和；`keys` 追加 (M) 指标；`summary` 调 `_task_summary("Mask")`。

### 类 `PoseMetrics`（第 1447～1589 行）

与分割对称：`**tp_p**`、`pose` Metric、Pose 前缀曲线与 summary。

### 类 `ClassifyMetrics`（第 1592～1670 行）

`process`：top-1 与 top-5 准确率；**fitness**=(top1+top5)/2；无 PR 曲线。

### 类 `OBBMetrics`（第 1673～1695 行）

继承 `DetMetrics`，匹配与 AP 流程同检测，验证器侧用 ProbIoU 填 `tp`。

### 类 `SemanticMetrics`（第 1698～1917 行）

**`update_stats`（1733～1755）**：忽略 255，累加混淆矩阵；统计每类出现图像数。

**`process`（1757～1792）**：对角交并比→**mIoU**（仅 GT 出现类平均）、像素准确率、per-class IoU/PA；可画 IoU 柱状图。

**属性**（1828～1887）：`miou`、`pixel_accuracy`、`fitness=miou`；`summary` 每类 IoU 与全局 mIoU。

### 类 `DepthMetrics`（第 1920～2099 行）

**`__init__`（1938～1960）**：深度范围与 **median 对齐**（尺度不变评估）。

**`update_stats`（1962～2005）**：Eigen 有效深度 mask；可选 `median(gt)/median(pred)` 缩放；**delta1/2/3** 为 `max(p/g,g/p)` 阈值比例；**abs_rel、rmse、silog** 按图算后 CPU float64 累加。

**`process`（2007～2023）**：图均值写入 `_results`。

**属性**（2031～2095）：`fitness=delta1`；`summary` 单行全局深度指标。