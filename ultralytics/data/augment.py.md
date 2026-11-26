# `ultralytics/data/augment.py`

本文件是 Ultralytics YOLO 训练与推理数据管线的**图像与标注联合增强**中心：把 NumPy/PIL 图像与 `Instances`、语义掩码、深度图、开放词汇文本等统一放进 `labels` 字典，经可组合变换后交给 `Format` 转为 PyTorch 张量供 `collate_fn` 批处理。对外暴露以 `BaseTransform` 为根的变换类、`Compose` 流水线、`v8_transforms` 检测训练默认组合，以及分类任务的 `classify_transforms` / `classify_augmentations`。主要调用方包括 `ultralytics/data/dataset.py`（`YOLODataset`、`ClassificationDataset` 等构建 `transforms`）、`data/build.py` 推理源预处理中的 `LetterBox`，以及部分解决方案/SAM 相关代码对 `LetterBox`、`Format` 的复用。

**通用约定**：`labels["img"]` 多为 `uint8`、形状 `(H, W, C)`、OpenCV **BGR**；`labels["instances"]` 为 `Instances`，训练早期多为 **归一化** 的 `xywh`（相对宽高），几何变换前常 `convert_bbox` + `denormalize(w, h)` 回到像素坐标；`labels["cls"]` 通常为形状 `(N, 1)` 的类别索引。

---

## 逐段说明

### 导入与模块常量（L1–L25）

- **标准库**：`math`/`random` 用于随机仿射与采样；`deepcopy` 用于 CopyPaste 拷贝实例；`typing.Any` 标注字典类型。
- **数值与视觉**：`cv2`、`numpy` 负责 resize、warp、LUT、轮廓等；`torch`/`F` 用于张量与插值；`PIL.Image` 供 `CenterCrop` 读入。
- **Ultralytics 内部**：`polygons2masks*` 把多边形转掩码；`Instances` 封装框/分割/关键点；`bbox_ioa` 算框交叠比；`segment2box`、`xywh2xyxy`、`xyxyxyxy2xywhr` 做框格式与 OBB 转换；`TORCHVISION_*` 门控分类 auto augment 版本。
- **`DEFAULT_MEAN` / `DEFAULT_STD`（L24–L25）**：分类管线 ImageNet 风格归一化的默认值 `(0,0,0)` 与 `(1,1,1)`。

---

### `BaseTransform`（L28–L121）

检测/分割/姿态/深度等任务的**变换抽象基类**：约定 `labels` 字典键与调用顺序。

#### `BaseTransform.__call__`（L45–L59）

依次 `get_params` → `apply_image` → `apply_instances` → `apply_semantic` → `apply_depth`，保证同一随机参数（如仿射矩阵）同步作用于图与标注。

#### `BaseTransform.get_params`（L61–L73）

默认返回空 dict；子类在此采样随机性并返回共享 `params`。

#### `BaseTransform.apply_image` / `apply_instances` / `apply_semantic` / `apply_depth`（L75–L121）

默认恒等返回；子类覆盖对应模态。

---

### `Compose`（L124–L276）

将多个可调用变换**顺序串联**；元素可以是 `BaseTransform` 或任意 `labels -> labels` 可调用对象。

#### `Compose.__init__`（L146–L152）

非 list 的单变换包装成长度为 1 的列表。

#### `Compose.__call__`（L154–L173）

`for t in self.transforms: data = t(data)`，前一步输出作为下一步输入。

#### `Compose.append` / `insert`（L175–L200）

列表尾部追加或指定下标插入。

#### `Compose.__getitem__`（L202–L221）

`int` 返回单个变换；`list` 返回新的 `Compose` 子链。

#### `Compose.__setitem__`（L223–L247）

支持 int 或等长 list 批量替换，带下标范围断言。

#### `Compose.tolist` / `__repr__`（L249–L276）

暴露底层 list；`repr` 打印变换名列表。

---

### `BaseMixTransform`（L279–L432）

Mosaic、MixUp、CutMix、CopyPaste（mixup 模式）的**公共骨架**：按概率 `p` 触发，从 `dataset` 拉额外样本，可选 `pre_transform`。

#### `BaseMixTransform.__init__`（L309–L322）

保存 `dataset`、`pre_transform`、`p`；`preserve_obb` 跟数据集 `use_obb` 决定裁剪时是否保留旋转框语义。

#### `BaseMixTransform.__call__`（L324–L349）

`random.uniform(0,1) > p` 则跳过；否则与 `BaseTransform` 类似四步 apply，最后 `pop("mix_labels")` 避免泄漏中间态。

#### `BaseMixTransform.get_params`（L351–L375）

`get_indexes()` 得索引列表；`dataset.get_image_and_label(i)` 组装 `mix_labels`；若有 `pre_transform` 对每个 mix 样本先跑一遍；写入 `labels["mix_labels"]` 并 `_update_label_text`。

#### `BaseMixTransform.get_indexes`（L377–L388）

默认随机一个整型索引；子类返回多索引。

#### `BaseMixTransform._update_label_text`（L390–L432）

开放词汇场景：无 `texts` 则返回。合并主图与 mix 图全部 `texts`，**去重保序**建 `text2id`；遍历各 label 把 `cls` 从「类内文本下标」重映射为**全局文本表下标**，并统一 `texts` 列表。

---

### `Mosaic`（L435–L781）

2×2（`n=4`）或 3×3（`n=9`）拼图；拼接中心 `(xc,yc)` 随机，输出大画布再裁回 `2*imgsz` 或中心 `imgsz` 区域。

#### `Mosaic.__init__`（L464–L482）

断言 `p∈[0,1]`、`n∈{4,9}`；`border=(-imgsz//2,-imgsz//2)` 用于 9 宫格裁剪；`buffer_enabled = (dataset.cache != "ram")` 时从数据集 buffer 采样以配合多 worker。

#### `Mosaic.get_indexes`（L484–L502）

需 `n-1` 个伴图索引：buffer 模式 `random.choices(buffer)`，否则全库 `randint`。

#### `Mosaic.get_params`（L504–L597）

**段 1 — 前置与 4 图模式随机中心（L513–L524）**  
`super().get_params` 已填入 `mix_labels`；若 `labels` 带 `rect_shape` 断言失败（矩形 batch 与 mosaic 不能同时开）。`s=self.imgsz`；`n==4` 时在扩展范围 `[-border, 2s+border]` 上均匀采样拼接中心 `(yc, xc)`，决定四张图在 `2s×2s` 画布上的交汇点。

**段 2 — 4 图模式象限几何（L525–L554）**  
循环 `i in 0..3`：`labels_patch` 首张用当前 `labels`，其余用 `mix_labels[i-1]`；`(h,w)` 来自 `resized_shape` 或 `img.shape`。按象限分支：左上/右上/左下/右下分别用 `max/min` 裁剪画布窗口 `x1a..y2a` 与源图子区 `x1b..y2b`，使交汇点两侧各贴一块；`padw=x1a-x1b`、`padh=y1a-y1b` 即实例坐标最终要加的平移量。每项写入 `layout` 字典列表。

**段 3 — 9 图模式与返回（L555–L597）**  
`hp,wp` 记录上一块高宽以铺邻接格；`i==0` 中心块锚定在 `(s,s)` 并记 `h0,w0`；其余 8 索引按论文式固定相对位移拼 `c=(x1,y1,x2,y2)`，再用 `max(x,0)` 得画布合法 ROI；`padw,padh=c[:2]` 供实例平移。每轮更新 `hp,wp=h,w`。最后 `params["layout"]=layout`。

#### `Mosaic.apply_image`（L599–L630）

`n==4`：`img4` 填充 114，按 layout 循环贴片。`n==9`：更大画布贴片后用 `border` 负索引裁成 `imgsz×imgsz` 视场。

#### `Mosaic.apply_instances`（L632–L655）

对每块 `_update_labels` 平移坐标后 `_cat_labels` 合并，再 `labels.update`。

#### `Mosaic.apply_semantic`（L657–L697）

无任一语义掩码则跳过；否则与图像同几何贴片，缺省填充 255（忽略类）。

#### `Mosaic._update_labels`（L699–L725）

用 `img_shape` 或当前 `img` 得 `nh,nw`；实例转 `xyxy`、反归一化、`add_padding(padw,padh)`——因 `apply_image` 可能已覆盖 `labels["img"]`。

#### `Mosaic._cat_labels`（L727–L781）

拼接 `cls` 与 `Instances.concatenate`；`resized_shape=(2*imgsz,2*imgsz)`；`clip` 到马赛克边界；`remove_zero_area_boxes` 同步过滤 `cls`；保留 `texts`（若有）。

---

### `MixUp`（L784–L883）

像素加权混合 + **标签集合并集**（非标签插值）。

#### `MixUp.__init__`（L808–L819）

仅调父类。

#### `MixUp.get_params`（L821–L832）

`r ~ Beta(32,32)`，混合权重偏向 0.5 附近。

#### `MixUp.apply_image`（L834–L847）

`img = img*r + img2*(1-r)` 再 `uint8`。

#### `MixUp.apply_instances`（L849–L862）

两图 `Instances` 与 `cls` 在实例维拼接。

#### `MixUp.apply_semantic`（L864–L883）

两图都有掩码时，若 `r<0.5` 整幅采用第二张掩码，避免分数类 ID。

---

### `CutMix`（L886–L1059）

在首张图上挖矩形贴入第二张对应区域，并追加与洞相交的实例。

#### `CutMix.__init__`（L913–L925）

`beta`、`num_areas` 控制候选洞大小与数量。

#### `CutMix._rand_bbox`（L927–L954）

`lam~Beta` → `cut_ratio=sqrt(1-lam)` 得 `cut_w,cut_h`；随机中心 `(cx,cy)`；`clip` 得 `xyxy` 像素框。

#### `CutMix.get_params`（L956–L987）

采样 `num_areas` 个候选洞；`bbox_ioa` 与主图框求交，剔除与任何 GT 重叠的洞；无可用则 `skip=True`。随机选一洞；对第二张算 `ioa2`，筛与洞重叠的实例（分割任务阈值 0.01，纯框 0.1）；无实例且无语义掩码则 skip。否则记下 `area`、`indexes2`、`w,h`。

#### `CutMix.apply_image`（L989–L1004）

非 skip 时将 `labels2` 矩形区域复制到主图。

#### `CutMix.apply_instances`（L1006–L1036）

取出 `instances2[indexes2]`，反归一化后平移到局部坐标、裁剪到洞尺寸，再平移回全图；拼 `cls` 与 `instances`；OBB 模式可能二次过滤零面积。

#### `CutMix.apply_semantic`（L1038–L1059）

复制第二张语义掩码的同一矩形到主掩码。

---

### `RandomPerspective`（L1062–L1454）

随机透视/仿射：旋转、缩放、平移、剪切、透视；同步 warp 图、框、多边形、关键点、语义掩码、深度。

#### `RandomPerspective.__init__`（L1099–L1130）

保存各幅度；`size` 为输出 `(w,h)`；`preserve_obb` 控制分割点是否严格 clip 在轴对齐框内。

#### `RandomPerspective._compute_affine_matrix`（L1132–L1175）

以图像中心为原点构建 `C,P,R,S,T` 五个 3×3 矩阵：`R` 含旋转角 `a` 与缩放 `s`（`scale` 可为区间元组）；`S` 用 `tan(shear°)`；`T` 平移与 `size` 成比例；`M = T@S@R@P@C`（右乘顺序重要）。

#### `RandomPerspective.get_params`（L1177–L1193）

`rect_shape` 优先定 `size`；否则用图宽高或 `self.size`；算 `M` 与 `scale`、`orig_shape`。

#### `RandomPerspective.apply_image`（L1195–L1217）

有透视用 `warpPerspective`，否则 `warpAffine(M[:2])`；边界填充 114；维护 `resized_shape`。

#### `RandomPerspective.apply_instances`（L1219–L1262）

**段 1 — 准备与框变换（L1229–L1237）**  
保存 `cls`；`pop` 出 `instances` 避免原地污染；转 `xyxy` 后用 `orig_shape` 的 `(w,h)` denormalize 到像素。取 `params["M"]`、`scale`，`apply_bboxes` 得仿射/透视后的 `bboxes`。

**段 2 — 分割/关键点与投影面筛选（L1239–L1252）**  
若有 `segments` 用之，否则把轴对齐框四角当作 8 点折线；`pts @ M[2,:2] + M[2,2]` 得每点透视除数 `w`，后续 `w.min*w.max>0` 剔除翻转退化的目标。有分割则 `apply_segments` 同步更新点与推导框；有关键点则 `apply_keypoints` 并可能把越界点 visibility 清零。组装 `new_instances` 并按 `size` clip。

**段 3 — 面积/宽高比门控（L1254–L1262）**  
对**变换前**实例仅 bbox 乘 `scale` 与 `new_instances` 对齐比较；`box_candidates` 要求新框不过小、面积比与长宽比合理（分割任务 `area_thr=0.01`，纯检测 0.10）。布尔索引 `i` 同时约束透视 `w` 符号与候选条件；写回过滤后的 `instances` 与 `cls[i]`。

#### `RandomPerspective.apply_bboxes`（L1264–L1296）

每框四角八点齐次乘 `M.T`，透视时除以 `w`；再 min/max 得新 `xyxy`。

#### `RandomPerspective.apply_segments`（L1298–L1326）

所有顶点仿射+透视除法；`segment2box` 得新框；非 OBB 时点坐标 clip 到框内防越界。

#### `RandomPerspective.apply_keypoints`（L1328–L1360）

变换 `x,y`；出界则 visibility 置 0。

#### `RandomPerspective.apply_semantic`（L1362–L1383）

掩码 `INTER_NEAREST` warp，边界 255。

#### `RandomPerspective.apply_depth`（L1385–L1410）

深度 `INTER_NEAREST` warp，边界 0，保持米制数值不被插值伪造。

#### `RandomPerspective.box_candidates`（L1412–L1454）

静态：比较变换前后框宽高、面积比、宽高比，布尔掩码保留合规模板框。

---

### `RandomHSV`（L1457–L1531）

仅改图像通道，不动标注。

#### `RandomHSV.__init__`（L1480–L1492）

`hgain/sgain/vgain` 为各通道随机扰动上界。

#### `RandomHSV.apply_image`（L1494–L1531）

非 3 通道跳过；随机 `r∈[-1,1]^3` 乘增益建 LUT；BGR→HSV，`cv2.LUT` 再 HSV→BGR 写回 `img`；`lut_sat[0]=0` 防止纯白偏色。

---

### `RandomFlip`（L1534–L1675）

水平或垂直翻转，概率 `p`。

#### `RandomFlip.__init__`（L1555–L1574）

断言方向与概率；`flip_idx` 供姿态关键点左右对称索引置换。

#### `RandomFlip.get_params`（L1576–L1596）

按 `p` 决定是否翻转；归一化实例用 `h=w=1` 作翻转参考。

#### `RandomFlip.apply_image`（L1598–L1615）

`flipud` 或 `fliplr`。

#### `RandomFlip.apply_instances`（L1617–L1637）

转 `xywh` 后 `flipud/fliplr`；关键点按 `flip_idx` 重排。

#### `RandomFlip.apply_semantic` / `apply_depth`（L1639–L1675）

对掩码/深度同样翻转。

---

### `LetterBox`（L1678–L1937）

保持纵横比的 resize + 填充到 `new_shape`，检测/分割/姿态主流程标准预处理。

#### `LetterBox.__init__`（L1705–L1739）

`auto` 按 stride 取余 padding；`scale_fill` 拉伸填满；`scaleup` 控制是否允许放大；`padding_value` 默认 114。

#### `LetterBox.__call__`（L1741–L1777）

可只传 `image` 返回 ndarray；否则完整走 image+instances；`return_image_only` 空 dict 时仅返回图。

#### `LetterBox.get_params`（L1779–L1826）

`rect_shape` 可覆盖 `new_shape`（矩形 batch 训练）；`r=min(new_h/h,new_w/w)` 且可选 cap 为 1；算 `new_unpad`、`dw,dh` 与四边 `top/bottom/left/right`；`center` 时分摊一半 padding。

#### `LetterBox.apply_image`（L1828–L1861）

`cv2.resize` 到 `new_unpad`；3 通道 `copyMakeBorder`，多光谱手写 pad 数组。

#### `LetterBox.apply_semantic`（L1863–L1884）

最近邻 resize + 边界 255。

#### `LetterBox.apply_instances`（L1886–L1905）

有实例则 `_update_labels`；累积 `ratio_pad` 供验证还原坐标。

#### `LetterBox._update_labels`（L1907–L1937）

`xyxy` + denormalize + `scale(ratio)` + `add_padding(left,top)`。

---

### `CopyPaste`（L1940–L2101）

实例分割 Copy-Paste：`flip` 模式贴镜像自身；`mixup` 模式从数据集取第二图。

#### `CopyPaste.__init__`（L1966–L1981）

`mode` 必须为 `flip` 或 `mixup`。

#### `CopyPaste.__call__`（L1983–L1993）

无分割或 `p==0` 跳过；`flip` 直接 get_params+apply 三件套；`mixup` 走父类概率门控。

#### `CopyPaste.get_params`（L1995–L2035）

`mixup` 时父类拉 `labels2`；主实例 denormalize；`labels2` 缺省则复制并 `fliplr`；`bbox_ioa` 找与已有实例 IoA 全 `<0.3` 的候选，按最大 IoA 排序取前 `round(p*len)` 为 `selected`；`im_new` 全零作粘贴掩码。

#### `CopyPaste.apply_image`（L2037–L2064）

在 `im_new` 上 `drawContours` 选中实例；源图默认水平翻转或 `labels2_img`；`cv2.copyTo` 只把源像素写入轮廓内。

#### `CopyPaste.apply_instances`（L2066–L2088）

拼接选中类的 `cls` 与 `instances2[selected]`。

#### `CopyPaste.apply_semantic`（L2090–L2101）

语义源为 mix 掩码或 flip 掩码，同样 `copyTo`。

---

### `Albumentations`（L2104–L2325）

可选第三方增强；初始化失败则 `transform=None` 静默降级。

#### `Albumentations.__init__`（L2131–L2202）

懒加载 `albumentations`；可从 checkpoint dict 还原；检测空间变换则 `BboxParams(yolo)` + `KeypointParams`；`topology_transforms` 设 deterministic；默认一组低概率 Blur/CLAHE 等。

#### `Albumentations.__call__`（L2204–L2325）

**段 1 — 门控与输入打包（L2238–L2273）**  
`transform is None` 或 `random.random()>=p` 原样返回。需要时 `set_random_seed(torch.initial_seed())` 对齐各 DataLoader worker。通道数非 1/3 跳过。空间模式：`instances` 转 `xywh` 并 normalize 到 `[0,1]`；把多边形顶点与关键点 `x,y` 拼成 `points` 再乘 `(w,h)` 变像素坐标；末尾可加三个锚点 `(0,0),(w,0),(0,h)` 供检测反射。调用 Albumentations 传入 `bboxes`、`class_labels`、`idx`、`keypoints`、`pidx`，可选 `topology` 字典与 `mask`。

**段 2 — 拓扑与点丢失修复（L2274–L2304）**  
若多边形/关键点存在且 `topology` 非空，直接 `NotImplementedError`（RandomGridShuffle 不保拓扑）。根据返回的 `pidx` 建 `lost` 掩码，未保留点保持原坐标，保留点用 `new["keypoints"]` 回填。分割分支对每实例缺失顶点用邻近可见点插值；关键点分支合并越界与丢失 visibility，并通过锚点叉积 `a×b` 判断是否镜像，需要时 `flip_idx` 重排。

**段 3 — 重建 Instances 与输出（L2305–L2325）**  
有分割：重塑 `segments`，逐实例 `segment2box` 得 `xyxy` 像素框并把点 clip 到框内，`Instances` 再 normalize；仅框检测：直接 `instances.update(new bboxes)` 并可 normalize 关键点。写回 `labels["img"]`、`labels["cls"][i]`、`labels[key]` 掩码/深度。无空间变换时一行 `self.transform(image=im)` 更新图像。

---

### `Format`（L2328–L2559）

训练末尾把 numpy 标注转为 tensor，供 DataLoader collate。

#### `Format.__init__`（L2360–L2397）

控制 `bbox_format`、是否归一化、mask/keypoint/obb、`mask_ratio` 与 overlap 策略、`batch_idx`、`bgr` 随机 RGB 翻转概率。

#### `Format.get_params`（L2399–L2418）

读 `h,w`；`pop` `cls`/`instances` 并转目标框格式+像素坐标。

#### `Format.apply_image`（L2420–L2433）

`_format_img` → CHW tensor。

#### `Format.apply_instances`（L2435–L2496）

**段 1 — 掩码分支与空实例（L2448–L2476）**  
`return_mask` 时若 `mask_ratio>min(h,w)` 抛错防零尺寸掩码。`nl>0` 调 `_format_segments` 得下采样 `(N,h',w')` 或 overlap 单通道索引掩码；转 `torch` 后算 `sem_masks`：overlap 模式用实例索引查类 ID，非 overlap 用面积权重最小覆盖规则合成语义图。无实例则填零张量占位。

**段 2 — 检测/姿态/OBB 张量（L2477–L2492）**  
`cls`、`bboxes` 从 numpy 转 tensor，空目标用 shape `(0,4)`/`(nl,1)` 占位。`return_keypoint` 写 `keypoints` 并在 `normalize` 时除 `w,h`。`return_obb` 用八点段转 `xywhr` 替换 `bboxes`。归一化轴对齐框时 `[0,2]` 除宽、`[1,3]` 除高。

**段 3 — collate 辅助字段（L2493–L2496）**  
`batch_idx` 为长度 `nl` 的全零向量，collate 时再叠真实 batch 内索引。

#### `Format._format_img`（L2498–L2526）

HWC→CHW，`bgr` 概率通道反转，`torch.from_numpy`。

#### `Format._format_segments`（L2528–L2559）

`mask_overlap` 用 `polygons2masks_overlap` 并排序实例；否则独立掩码列表。

---

### `SemanticFormat`（L2562–L2600）

语义分割专用：`apply_image` 额外把 `semantic_mask` 转 `int32` tensor；`apply_instances` 删掉检测相关键，保留 `ori_shape`/`ratio_pad` 便于验证可视化。

---

### `LoadVisualPrompt`（L2603–L2711）

为多模态/提示学习生成按类别的低分辨率视觉掩码。

#### `LoadVisualPrompt.__init__`（L2606–L2612）

`scale_factor` 默认 1/8，相对输入 CHW 图高宽。

#### `LoadVisualPrompt.make_mask`（L2614–L2630）

向量化 `xyxy` 填充满 `(N,h,w)` 二值掩码。

#### `LoadVisualPrompt.get_params`（L2632–L2650）

从已有 tensor `bboxes`（xywh 反归一化到 xyxy 像素）或 `masks` 取几何；`cls` 展平为 int。

#### `LoadVisualPrompt.apply_image`（L2652–L2664）

`get_visuals` 结果写入 `labels["visuals"]`。

#### `LoadVisualPrompt.get_visuals`（L2666–L2711）

缩放框或最近邻缩掩码；按 `torch.unique(category)` 通道聚合 `logical_or`，输出 `(num_classes, H', W')`。

---

### `RandomLoadText`（L2714–L2832）

开放词汇检测：每步采样正负类文本，重映射 `cls` 并重写 `texts` 列表。

#### `RandomLoadText.__init__`（L2739–L2766）

`prompt_format`、`neg_samples` 范围、`max_samples`、是否 padding 空白文本。

#### `RandomLoadText.get_params`（L2768–L2817）

正类去重，超限则 random sample；负类数 `randint(*neg_samples)`  capped；建 `label2ids`；过滤不在采样表中的实例得 `valid_idx` 与新 `cls`；每类随机选一条 prompt 字符串；padding 补到 `max_samples`。

#### `RandomLoadText.apply_instances`（L2819–L2832）

实例子集、`cls`、`texts` 写回 labels。

---

### `v8_transforms`（L2835–L2916）

**YOLOv8 检测训练默认流水线工厂**。

- 建 `Mosaic` + `RandomPerspective(size=imgsz)` 为 `affine`；`pre_transform = Compose([mosaic, affine])`。
- `copy_paste_mode=="flip"`：在索引 1 插入 `CopyPaste`；否则在末尾 append 带独立 `pre_transform` 的 CopyPaste（mixup 模式）。
- 关键点数据检查 `flip_idx` 与 `kpt_shape`，缺失则警告并关掉 flip 增广。
- 返回 `Compose([pre_transform, MixUp, CutMix, Albumentations, RandomHSV, flipud, fliplr])`。

---

### `classify_transforms`（L2920–L2966）

**分类验证/推理**：torchvision `Resize`（方形象用标量最短边）、`CenterCrop`、`ToTensor`、`Normalize`；废弃 `crop_fraction` 会 `deprecation_warn`。

---

### `classify_augmentations`（L2970–L3070）

**分类训练**：`RandomResizedCrop` + 可选水平/垂直翻转；`auto_augment` 分支按 torchvision 版本挂 RandAugment/AugMix/AutoAugment；可选 `ColorJitter`（与 AA 互斥除非 `force_color_jitter`）；末尾 Normalize + `RandomErasing`。

---

### `DepthFormat`（L3073–L3112）

深度任务：`apply_depth` 把深度 resize 到与 `img` tensor 同 `(H,W)`，输出 `(1,H,W) float`；`apply_instances` 剔除检测字段。

---

### `ClassifyLetterBox`（L3116–L3187）

**向后兼容**分类用简易 letterbox：仅处理 `ndarray` 输入输出，无 labels；`auto` 时按 stride ceil 目标尺寸；居中 pad 114。

#### `ClassifyLetterBox.__call__`（L3156–L3187）

算缩放比 `r`、resize 嵌入 `im_out`。

---

### `CenterCrop`（L3191–L3248）

**向后兼容**：最大内接正方形中心裁切再 resize 到 `(w,h)`；支持 PIL 转 numpy。

---

### `ToTensor`（L3252–L3313）

**向后兼容**：HWC BGR numpy → CHW tensor，可选 `half`，除以 255 归一化到 [0,1]。

---

## 文件边界小结

| 概念 | 含义 |
|------|------|
| `labels` | 管道中传递的字典，键随任务扩展 |
| 像素坐标 | 相对当前 `labels["img"]` 宽高，原点左上，x 向右 y 向下 |
| 归一化框 | 相对宽高的比例，常见于数据集刚读出时 |
| Mosaic 画布 | `2*imgsz` 或 `3*imgsz` 临时坐标，裁切后落入 `imgsz` 训练尺寸 |