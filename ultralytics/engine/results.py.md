# `ultralytics/engine/results.py`

本模块定义推理结果的**容器与张量包装类型**：`Results` 聚合单张图上的检测/分割/分类/姿态/OBB/语义/深度输出；`Boxes`、`Masks`、`Keypoints`、`OBB`、`Probs` 等对原始张量提供坐标变换与属性访问；`BaseTensor` 统一设备迁移。主要由 `engine/predictor.py` 在 postprocess 后构造 `Results` 列表返回；用户通过 `model.predict()` / `model()` 得到的可迭代对象即此类实例。`Results` 还混入 `DataExportMixin`（定义在 `ultralytics.utils`，本文件无源码）提供 `to_df` / `to_csv` / `to_json`，内部调用本类的 `summary()`。可视化走 `plot`/`save`，标注逻辑依赖 `utils.plotting.Annotator`。

## 第 8～20 行：导入分组

- `bisect_left`：`Masks.crops` 双线性缩放时定位 letterbox 映射区间。
- `deepcopy`, `cached_property`, `Path`, `Any`：绘图拷贝、惰性属性、路径与类型。
- `numpy`, `torch`：底层数组与张量。
- `LOGGER`, `DataExportMixin`, `SimpleClass`, `ops`：日志、mixin、轻量基类、几何与 mask 算子。
- `FONT`, `Annotator`, `colors`, `save_one_box`：绘制与裁剪。

## 逐段说明

### 第 23～177 行：`BaseTensor`

**职责**：在 `data`（`torch.Tensor | np.ndarray`）与 `orig_shape (H,W)` 上封装预测张量，子类继承设备与索引语义。

**`__init__`（50～59）**：断言 `data` 类型；保存 `data`、`orig_shape`。

**`shape`（61～74）**：属性，返回 `self.data.shape`。

**`_convert(fn)`（76～78）**：对新 `data` 应用 `fn`，同 `orig_shape` 构造同类新实例——`cpu`/`numpy`/`cuda`/`to` 共用。

**`cpu`（80～95）**：ndarray 不变，否则 `.cpu()`。

**`numpy`（97～111）**：已是 ndarray 则保持，否则 `cpu().numpy()`。

**`cuda`（113～128）**：`torch.as_tensor(x).cuda()`。

**`to`（130～145）**：`as_tensor` 后 `.to(*args, **kwargs)`。

**`__len__`（147～159）**：`len(self.data)`，通常为第 0 维实例数。

**`__getitem__`（161～177）**：索引 `data` 后包装为新 `BaseTensor`，`orig_shape` 不变（逐检测切片）。

### 第 180～191 行：`_DenseResultTensor`

语义/深度等**整图一张图一个结果**，行索引不应把 H×W 当实例维度。

**`__len__`（183～185）**：`data.shape[0] > 0` 则逻辑长度为 1，否则 0（空图）。

**`__getitem__`（187～191）**：tensor 索引先转 numpy；用 `np.arange(len(self))[idx]` 判断是否空选；空则 `data[:0]`，否则原 `data`——避免对语义图做“第 i 个检测”式切片。

### 第 194～199 行：`SemanticMask` / `DepthMap`

空子类，仅类型标签；语义为 (H,W) 或 (1,H,W) 类别 id 图；深度为 (H,W) 米制深度。

### 第 202～963 行：`Results`

#### 第 253～304 行：`__init__`

- 保存 `orig_img`、`orig_shape = shape[:2]`。
- 各任务张量非空则包装：`Boxes`/`Masks`/`Probs`/`Keypoints`/`OBB`/`SemanticMask`/`DepthMap`。
- `speed` 默认三段 ms 占位；`names`、`path`；`colors`/`save_dir` 初值为 None。
- `_keys` 元组列出可批量 `apply` 的属性名。

#### 第 306～339 行：`merge_tiles`（类方法）

- 输入多 tile 的 `Results` 与 `windows`、`orig_img`、NMS 参数。
- 若有 `obb` 则 `rotated=True`，从各 result 收集 `local` 或 `.data` 到 `predictions`；可选 `feats`。
- boxes/obb 张量拆成 `bboxes`、`conf`、`cls` 列。
- `ops.merge_tiles` 合并到整图尺度；再拼回 `boxes`/`obb` 列；重建 `cls`；`feats`、`_loose` 可选递归合并。  
**调用方**：大图滑窗 predict 的后处理。

#### 第 341～355 行：`__getitem__`

- `_apply("__getitem__", idx)`：对每个非空 `_keys` 子张量同样索引，得到新 `Results`（支持 int/slice）。

#### 第 357～373 行：`__len__`

- 按 `_keys` 顺序找第一个非空属性，返回其 `len`；全空则 0（检测数或分类为 1 等由子类决定）。

#### 第 375～419 行：`update`

- 传入非空则重建对应包装；`boxes` 经 `ops.clip_boxes` 限制在原图内。

#### 第 421～446 行：`_apply`

- `new()` 空壳，对每个非空 key 调用 `getattr(v, fn)(*args)`，用于 `cpu`/`cuda`/`numpy`/`to`/`__getitem__`。

#### 第 448～513 行：`cpu` / `numpy` / `cuda` / `to`

- 均 `_apply` 对应 `BaseTensor` 方法名。

#### 第 515～527 行：`new`

- 复制 `orig_img/path/names/speed/save_dir/colors`，预测张量为空，供 `_apply` 填。

#### 第 529～667 行：`plot`（>30 行，分三段）

**参数与初始化（580～598）**：`color_mode` 断言；orig 为 tensor 时转 BGR uint8；按任务选 `pred_boxes`（OBB 用 `obb`）、masks、probs；需要时 boxes 一次 `.cpu()`；`Annotator` 深拷贝底图。

**绘制分支（600～657）**：  
- masks：`crops` 得 box-local 或整 mask；颜色按 track id / class / 序号；`annotator.masks`。  
- boxes：OBB 用 `xyxyxyxy`，否则 `xyxy`；逆序遍历画框与标签。  
- probs：top5 文字块。  
- semantic_mask：单类时 1→0 可视化；`semantic_mask` 半透明。  
- depth：`depth_map` 热力叠加。  
- keypoints：逆序 `kpts`，track 着色。

**输出（659～667）**：`show`/`save` 委托 annotator；返回 BGR ndarray 或 PIL。

#### 第 669～714 行：`show` / `save`

- `show`：`plot(..., show=True)`。
- `save`：默认文件名 `results_{stem}.jpg`，`plot(..., save=True)` 返回路径。

#### 第 716～750 行：`verbose`

- 无实例：`probs` 任务返回空串，否则 `"(no detections), "`。
- 分类：top5 名与概率。
- 有 boxes/obb：`cls` bincount 各类计数。
- 仅 depth：有效像素 min-max 米。

#### 第 752～822 行：`save_txt`

- 语义：警告不支持。  
- 分类：每行 `conf class_name`（top5）。  
- 检测：归一化 `xywhn` 或 OBB `xyxyxyxyn`；可选 mask 多边形 `xyn`（<3 点跳过）；pose 拼 `xyn`+可见度；`save_conf`、track `id`；追加写 UTF-8。

#### 第 824～862 行：`save_crop`

- 非检测类警告返回；遍历 boxes/OBB，`save_one_box` 到 `save_dir/class/file.jpg`。

#### 第 864～963 行：`summary`

- **probs**：top5 字典列表 `name/class/confidence`。  
- **semantic_mask**：各类像素占比；二值单类跳过背景 id。  
- **检测等**：无 boxes 且无 obb 返回 `[]`；否则每行 `name/class/confidence/box`（四点或八点坐标，可选 normalize）；track `track_id`；有 mask 加 `segments`；有 kpts 加 `keypoints`（含可选 `visible`）。  
**供** `DataExportMixin.to_df/to_json/to_csv` 消费。

#### 继承的导出方法（不在本文件）

- `to_df` / `to_csv` / `to_json`：`ultralytics.utils.DataExportMixin`，调用 `summary()` 转 Polars/字符串。

### 第 966～1154 行：`Boxes`

**`data` 行布局（核心字段）**：

| 列（6 列） | 含义 |
|------------|------|
| 0～3 | `xyxy` 像素坐标 x1,y1,x2,y2（相对 `orig_shape` 原图尺寸） |
| -2 | `conf` 目标置信度 |
| -1 | `cls` 类别 id（float 存储，用时转 int） |

**7 列（跟踪）**：在 conf 前多一列 `id`（track_id），即 `[-3]` 为 id，`[-2]` conf，`[-1]` cls。

**`__init__`（1002～1020）**：1D 升维；`n in {6,7}`；`is_track = (n==7)`。

**属性**：`xyxy` `[:,:4]`；`conf` `[:,-2]`；`cls` `[:,-1]`；`id` 仅 track 时 `[:,-3]`。

**`xywh`（1094～1109）**：`ops.xyxy2xywh`。

**`xyxyn` / `xywhn`（1111～1154）**：除以 `orig_shape` 宽高做 [0,1] 归一化。

### 第 1157～1321 行：`Masks`

**`data` 形态**：

- 密集：`(N, H, W)` 二值/概率 mask，与原图或 letterbox 空间对齐。  
- **box-local**（构造传 `tuple`）：`(rois, masks, shape)`，`rois` 每行 `(x1,y1,x2,y2, batch, mask_idx)`；惰性 `data` 首次访问时贴回全图。

**`__init__`（1183～1196）**：tuple 存 `local`；2D 升维 3D。

**`shape`（1198～1201）**：local 时为 `(len(rois), *shape)`。

**`data`（1203～1211）**：循环 `crops()` 贴到全零画布。

**`_convert`（1213～1220）**：local 时只变换各 crop mask 与 rois。

**`crops(shape)`（1222～1256）**：yield `(x1,y1,x2,y2), mask_crop`；若给定与当前不同的 `shape`，用 letterbox 映射 + 双线性 `lerp` 重采样到目标分辨率，阈值 >0.5。

**`__str__`/`__len__`/`__getitem__`**：local 语义与索引子集。

**`_segments`（1275～1279）**：轮廓点 + 平移到 box 原点。

**`xyn` / `xy`（1281～1321）**：轮廓经 `scale_coords` 到原图像素或归一化坐标列表（每 mask 一个 (M,2) ndarray）。

### 第 1324～1425 行：`Keypoints`

**`data` 形态**：`(N, K, 2)` 仅 x,y；或 `(N, K, 3)` 第三维为关键点可见度/置信度。

**`__init__`（1355～1369）**：2D→3D；`has_visible = (最后一维==3)`。

**`xy`（1371～1390）**：`[..., :2]` 像素坐标。

**`xyn`（1392～1410）**：除以原图宽高归一化。

**`conf`（1412～1425）**：有 visible 维时 `[..., 2]`，否则 `None`。

人体默认 17 点索引见 `Results.__init__` docstring（0 鼻、5/6 肩等）。

### 第 1428～1538 行：`Probs`

**`data` 形态**：`(num_classes,)` 分类 softmax 输出，**无**几何 `orig_shape` 语义（仍传 `BaseTensor`，可为 `None`）。

**`top1`（1474～1486）**：`argmax` 类别下标。

**`top5`（1488～1500）**：降序 argsort 取前 5 索引。

**`top1conf` / `top5conf`（1502～1537）**：对应概率值（`top5` 为索引数组花式索引）。

### 第 1541～1743 行：`OBB`

**`data` 行布局（7 列）**：

| 列 | 含义 |
|----|------|
| 0～4 | `xywhr`：中心 x,y、宽 w、高 h、旋转角 r（弧度，非训练用 long-edge 规范） |
| -2 | `conf` |
| -1 | `cls` |

**8 列（跟踪）**：`id` 在 `[-3]`，与 `Boxes` 相同尾部约定。

**`xywhr`（1596～1616）**：`[:, :5]`。

**`conf` / `cls` / `id`（1618～1670）**：同 Boxes 列索引逻辑。

**`xyxyxyxy`（1672～1687）**：`ops.xywhr2xyxyxyxy` → `(N, 4, 2)` 四角逆时针。

**`xyxyxyxyn`（1689～1707）**：四角坐标除以原图宽高。

**`xyxy`（1709～1743）**：四角 min/max 得轴对齐外接矩形，便于与非旋转 IoU/裁剪。

---

## 张量字段对照总表

| 类型 | `data` 形状要点 | 主要属性 | 含义 |
|------|-----------------|----------|------|
| **Boxes** | (N,6) 或 (N,7) | xyxy, conf, cls, id?, xywhn | 水平框检测 + 可选跟踪 |
| **Masks** | (N,H,W) 或 local tuple | data, xy, xyn | 实例分割 mask / 轮廓 |
| **Keypoints** | (N,K,2\|3) | xy, xyn, conf? | 姿态关键点 |
| **OBB** | (N,7\|8) | xywhr, xyxyxyxy, conf, cls, id? | 旋转框 |
| **Probs** | (C,) | top1, top5, top*conf | 图像级分类 |
| **SemanticMask** | (1,H,W) 或 (H,W) | 继承 Dense 索引语义 | 每像素类别 id |
| **DepthMap** | (H,W) | 米制深度 | 稠密深度 |

## 调用链简述

`Predictor.postprocess` → 构造 `Results(orig_img, path, names, boxes=..., ...)` → 返回 `list[Results]`；`validator`、solutions 与 notebook 示例直接读写上述属性。跟踪模式在 predictor 侧把 6 列 boxes 扩为 7 列后再包 `Boxes`/`OBB`。