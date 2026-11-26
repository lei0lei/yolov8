# `ultralytics/utils/plotting.py`

本模块提供训练/验证/预测可视化：Ultralytics 标准色板、`Annotator` 在 OpenCV 与 PIL 上绘制框/掩码/姿态/深度，以及 mosaic、曲线、调参结果、LayerCAM 等图。深度着色用内嵌 Spectral_r LUT 避免运行时依赖 matplotlib。文本路径支持 RTL 阿拉伯字形与 Unicode 字体缓存。公开 API 含 `colors` 单例、`plot_images`（默认后台线程）、`plot_results` 与 `class_activation_map` 等。

## 逐段说明

### `_gaussian_filter1d`（第 27～43 行）

构造高斯核，`symmetric` pad 后 `convolve` valid 段，替代 scipy 一维平滑。

### 类 `Colors`（第 46～211 行）

`__init__`：20 色 hex 转 RGB `palette`，构建 `pose_palette` uint8 数组。`__call__`：int 索引，优先 `custom[i]` 否则 `palette[i%n]`，`bgr` 时反转通道。`hex2rgb` 解析 `#rrggbb`。`parse`：None/空→None；字符串 `literal_eval` 为列表；全非 `#rrggbb` hex 则映射 RGB，否则 warning 返回 None。

### `_spectral_lut` / `colorize_depth`（第 236～289 行）

锚点线性插值 256 色 LUT 转 BGR。`colorize_depth`：无效深度≤0；disparity 模式用 1/d 与 2/98 分位；metric 用 min/max；clip 归一化后 `applyColorMap`，spectral 用自定义 LUT，无效像素置黑。

### `_arabic_forms` / `_rtl`（第 292～317 行）

Pillow 有 raqm 则 None；否则建阿拉伯展示形字典。`_rtl` 对非 ASCII 连接字形并反转 RTL，数字拉丁 run 保持阅读序。

### `_font` / `_text_mask` / `_label_patch`（第 320～347 行）

`check_font` 加载 TTF，Pillow≥9.2 补丁 `getsize`；光栅化灰度 alpha；缓存 RGB 标签底+字色块。

### 类 `Annotator`——`__init__`（第 350～449 行）

据 PIL/非 ASCII/example 决定是否 `pil`。PIL 图转 BGR numpy；张量需 HWC uint8。线宽默认 ~0.3% 平均边长；灰度扩通道、多通道裁前三。PIL 模式 BGR 副本+Unicode 字体；cv2 模式要求连续缓冲区。COCO 骨架与 pose 调色 BGR  limb/kpt。

### `Annotator.get_txt_color`（第 451～471 行）

显式 `txt_color` 优先；否则背景 RGB 算 luma>150 用深蓝 #111f68 否则白。

### `Annotator.box_label`（第 473～525 行）

转 numpy 列表；多边形 `polylines` 否则 `rectangle`；标签在框外上方或框内，PIL 用 `_label_patch` 或 cv2 填底色+putText；防越界右缘。

### `Annotator.masks`（第 527～566 行）

numpy：逐掩码着色 `addWeighted`。torch：全图用 `scale_masks`+分 band amax 混 alpha；local 列表仅框内区域混色。

### `Annotator.semantic_mask` / `depth_map`（第 568～603 行）

整图类索引 LUT 着色，`ignore_index` 保留底图；深度 `colorize_depth` 尺寸对齐后 blend。

### `Annotator.kpts`（第 605～664 行）

17 点 COCO 才画骨架；置信度或 (0,0) 跳过；圆点与 limb `line`。

### `Annotator.rectangle` / `text` / `_draw`（第 666～713 行）

双次矩形填/描边；PIL 多行 `_text_mask`+alpha 底；cv2 可选底色。`_draw` 裁剪 ROI 粘贴或 alpha 混合。

### `Annotator._to_numpy` / `result` / `show` / `save` / `get_bbox_dimension`（第 715～761 行）

张量落 CPU numpy；`result` 可 PIL RGB；Colab/Kaggle `display`，否则 `Image.show`；`imwrite` BGR；静态方法返回宽高面积。

### `plot_labels`（第 764～821 行）

`TryExcept`+`plt_settings`；Polars DataFrame；四类子图：类直方图上色、500 框示意图、中心 xy/wh 2d hist；保存 labels.jpg，`on_plot` 回调。

### `save_one_box`（第 824～885 行）

OBB 四角仿射拉正再 crop。`xywh` 可加 gain/pad、square 化、`clip_boxes` 切片；JPEG 用 PIL 无 chroma 二次采样；`BGR` 控制返回通道序。

### `plot_images`（第 888～1088 行）

`@threaded` 默认异步。labels 张量转 numpy；解析 cls/bboxes/conf/masks/kpts/semantic/depth；反归一化、通道处理、BGR mosaic 网格；可选缩放；`Annotator` 逐子图文件名、框（obb 转四角）、仅类名、关键点、实例 index 掩码 LUT、语义/深度子 Annotator；`save` 则 PIL 写盘否则返回 ndarray。

### `plot_results`（第 1091～1147 行）

glob `results*.csv`；Polars 读入；列名含 loss/metric 分两行子图；每列原始点+高斯平滑虚线；存 results.png。

### `plot_multitrain_results`（第 1150～1180 行）

柱状各数据集指标，橙色虚线均值，旋转 x 标签，存 multitrain_results.png。

### `plt_color_scatter`（第 1183～1212 行）

2D 直方图给每点着色后 scatter。

### `plot_depth_panels`（第 1215～1274 行）

每行 RGB+可选 GT 深度+多预测列；有 GT 时共享 vmin/vmax；`colorize_depth` metric 模式；可选标题条 cv2 写字。

### `plot_tune_results`（第 1277～1378 行）

读 NDJSON records；可选剔零 fitness；3σ 下界迭代滤离群；网格子图 hyperparam vs fitness 着色 scatter 标最佳+号；第二图 fitness 迭代曲线与分数据集初始/最佳对比。

### `class_activation_map`（第 1381～1492 行）

head 注册 pre-hook 抓特征、hook 抓类 logits（含 end2end one2one/one2many、RT-DETR、分类/分割/深度分支）。`enable_grad` 前向；可选 classes 过滤；keep 由 conf 与 top1 保证；最多 topk 次反向 LayerCAM：正梯度×激活、归一化、层间 max；叠 JET 热图存盘；返回 detach 后 preds。