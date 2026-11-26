# `ultralytics/data/dataset.py`

本文件实现 Ultralytics **按任务划分的数据集类**：YOLO 检测/分割/pose/OBB（`YOLODataset`）、深度、语义（PNG 或 polygon 栅格）、多模态文本、Grounding JSON、`ConcatDataset` 拼接，以及独立的 **`ClassificationDataset`**（torchvision ImageFolder 包装）。公共机制：**标签缓存** `*.cache`（版本 `DATASET_CACHE_VERSION="1.0.13"`）、多线程 `verify_*` 扫描、`build_transforms` 链（`v8_transforms` + `Format`/`SemanticFormat`/`DepthFormat`）、`collate_fn` 堆叠 batch。`build.build_yolo_dataset` 按 `cfg.task` 实例化此处类；`Trainer` 在 epoch 末可对 `close_mosaic` 关闭 mosaic。

**YOLO 文本标签格式**：每对象一行 `class cx cy w h`（归一化 xywh）；分割额外多边形点；pose 含关键点维。缓存 label dict 见 `result_to_label`。**语义 mask**：像素为 class id，`255` 忽略；`label_mapping` YAML 经 256 LUT 映射。

## 逐段说明

### 导入与常量（第 3–64 行）

- **`augment`**：`Compose`、`Format`、`v8_transforms`、`classify_*` 等。
- **`utils`**：`verify_image_label`、`img2label_paths`、`polygons2masks_overlap`、`DATASET_CACHE` 读写等。
- **`DATASET_CACHE_VERSION`**： bump 时强制全量重扫标签。

---

### `YOLODataset`（第 67–484 行）

`format_class = Format`；`task` 决定 `use_segments`/`use_keypoints`/`use_obb`；pose 须合法 `data['kpt_shape']`。

#### `__init__`（第 102–124 行）

校验 kpt_shape；`super().__init__(..., channels=data.get("channels",3))`。

#### `cache_labels`（第 126–171 行）

骨架：`verify_args()` → 线程池 `imap`；`result_to_label` 累加 nm/nf/ne/nc；有标签则 `save_dataset_cache_file(..., DATASET_CACHE_VERSION)`；训练无标签 `ValueError`，验证仅警告。

#### `get_label_files`（第 173–180 行）

`img2label_paths(im_files)` → 与图同 stem 的 `.txt`。

#### `get_cache_hash`（第 182–191 行）

`label_files + im_files + scan_args` 字符串哈希；语义背景类不算入 nc 计数规则见注释。

#### `scan_summary`（第 193–205 行）

`nf images, nm+ne backgrounds, nc corrupt`。

#### `verify_args`（第 207–223 行）

返回 `verify_image_label, zip(im_files, label_files, prefix, use_keypoints, nc, nkpt, ndim, single_cls)`。

#### `result_to_label`（第 225–249 行）

成功时 label：`cls`=`lb[:,0:1]`，`bboxes`=`lb[:,1:]` xywh 归一化，`segments`/`keypoints` 原样。

#### `verify_labels`（第 251–277 行）

segment/OBB/语义要求 box 数=segment 数，否则 `ValueError` 或清空 segments；全空 cls 警告。

#### `_load_or_scan_cache`（第 279–295 行）

加载 cache 须 version+hash 匹配；失败则 `cache_labels` 重建。

#### `get_labels`（第 297–328 行）

cache 旁路 `.cache`；rank0 打印扫描摘要；空 labels `RuntimeError`；更新 `im_files` 为缓存内路径；`verify_labels`。

#### `build_transforms`（第 330–361 行）

训练：`rect` 时 mosaic/mixup/cutmix 置 0；`v8_transforms`；语义且增广可能开 `use_segments` 供 copy_paste。验证：仅 `LetterBox(scaleup=False)`。末尾 **`Format`**：`return_mask/keypoint/obb`、`batch_idx=True`、`mask_ratio`/`overlap_mask`、`bgr` 随机通道仅训练。

#### `build_text_transforms`（第 363–384 行）

训练时在 Format 前插入 `RandomLoadText(max_samples≤80, padding_value=低频负类文本)`。

#### `_get_neg_texts`（第 386–390 行）

`threshold = min(max(freq), 100)`，取频次≥阈的类名作负样本池。

#### `close_mosaic`（第 392–402 行）

`mosaic/copy_paste/mixup/cutmix=0`，重建 transforms。

#### `update_labels_info`（第 404–454 行）

弹出 bbox/segments 等到 **`Instances`**：

- **Tile**：先 denormalize+转 xyxy 筛与窗口相交的实例；segments/keypoints 同步；`crop(tile)` 后再 normalize 到 tile 尺寸。
- **Resample segments**：OBB 用 100 点，否则 1000（或更长多边形自适应）；空则 `(0, resamples, 2)` 零数组。
- 输出 `label["instances"]`。

#### `collate_fn`（第 456–484 行）

键排序一致；`img`/`depth`/`semantic_mask` 等 `stack`；`visuals` pad_sequence；`masks/bboxes/cls/...` `cat` 维 0；`batch_idx` 各图加 offset i 再 cat — 供 `build_targets` 区分图内索引。

---

### `DepthDataset`（第 487–601 行）

`format_class = DepthFormat`；深度图 `images/→depth/` 同 stem `.png` 或 `.npy`；`depth_scale` 默认 1000。

#### `_depth_path_for`（第 499–507 行）

路径段 `images` 换 `depth`，优先 png 否则 npy。

#### `get_label_files` / `get_cache_hash` / `scan_summary` / `verify_args` / `result_to_label`（第 509–554 行）

`depth_files` 映射；`verify_image_depth`；label 无框空张量。

#### `verify_labels`（第 556–557 行）

空实现。

#### `_load_depth`（第 559–574 行）

`load_depth`；tile 时缓存全图深度、窗口 crop；尺寸不对则 NEAREST resize 到 `source_shape`。

#### `get_image_and_label`（第 576–584 行）

super 后 `depth` resize 到 `resized_shape` 与 RGB 对齐。

#### `build_transforms`（第 586–601 行）

禁用 mosaic 等；验证把 `LetterBox` 换 **`scale_fill=True`** 拉伸。

---

### `YOLOMultiModalDataset`（第 604–669 行）

#### `update_labels_info`（第 620–634 行）

super 后 `texts` = 每类 `names` 按 `/` 拆同义词列表。

#### `build_transforms`（第 636–645 行）

`build_text_transforms(..., data["nc"])`。

#### `category_names` / `category_freq`（第 647–669 行）

名称集合与训练集中 cls 频次（无标注时各类 freq=0 作负样本）。

---

### `GroundingDataset`（第 672–890 行）

COCO 式 JSON + caption span；`cache_labels` **覆写**不走 txt verify。

#### `__init__`（第 690–703 行）

`task in detect/segment`；`data={"channels":3}` 占位。

#### `get_img_files`（第 705–709 行）

`fraction=1.0` 全扫目录；`scan_files` 进 hash。

#### `get_cache_hash`（第 711–713 行）

`json + scan_files + segments标志`。

#### `_verify_instance_counts`（第 715–729 行）

已知混合集 JSON 硬编码实例总数 assert。

#### `cache_labels`（第 731–840 行）

读 JSON `images`/`annotations`；按图聚合：

- `iscrowd` 跳过；bbox xywh→中心 xywh 归一化；零面积跳过。
- `tokens_positive` 切 caption 得 `cat_name`；`cat2id`/`texts` 列表。
- 去重 `seen`；分割多边形 merge 或 bbox 转四角 segment；`segments2boxes` 更新框。
- 警告非 polygon RLE 丢弃。

`save_dataset_cache_file` 带 hash。

#### `get_labels`（第 842–863 行）

`.json.cache`；无 texts 抛错；`im_files` 更新。

#### `build_transforms`（第 865–874 行）

`build_text_transforms(..., max_samples)`。

#### `category_names` / `category_freq`（第 876–890 行）

从 label `texts` 统计。

---

### `YOLOConcatDataset`（第 893–929 行）

`ConcatDataset` + **`collate_fn` 委托 YOLODataset**；`close_mosaic` 递归子集。

---

### `SemanticDataset`（第 932–1164 行）

`format_class = SemanticFormat`；PNG mask 与图对齐；`label_lut`/`inverse_lut` 256 项。

#### `__init__`（第 954–969 行）

解析 mapping；`masks={}` 与 mosaic buffer 同步淘汰。

#### `update_labels`（第 971–985 行）

`include_class` 过滤像素类；nc=1 时忽略 classes 过滤警告；含 `bg_class_idx`。

#### `_parse_label_mapping` / `_build_label_luts`（第 987–1015 行）

`ignore_label`/`None`→255；forward/inverse LUT。

#### `get_label_files`（第 1017–1028 行）

`masks_dir` 默认 `masks`；无 png 试其他 `IMG_FORMATS` 后缀。

#### `get_cache_hash`（第 1030–1037 行）

含 `label_mapping` JSON 与 `nc`。

#### `verify_args` / `result_to_label`（第 1043–1067 行）

`verify_image_mask` + `invalid` LUT 掩码；label 带 `mask_file`/`mode`。

#### `verify_labels`（第 1069–1071 行）

同步 `mask_files` 列表。

#### `get_labels`（第 1073–1088 行）

`single_cls` 不支持。

#### `load_image`（第 1090–1092 行）

训练 **短边缩放到 imgsz**（`resize_short=True`）。

#### `load_mask` / `_read_mask`（第 1094–1123 行）

tile 共享 `_source_mask`；`read_mask` + 可选 LUT。

#### `convert_label`（第 1125–1136 行）

`cv2.LUT` 或 numpy 索引。

#### `get_image_and_label`（第 1138–1164 行）

附加 `semantic_mask`；`include_class` 外像素→255；与图 NEAREST 对齐；buffer 索引缓存 `masks` dict 限长。

---

### `PolygonSemanticDataset`（第 1167–1218 行）

无 `masks_dir` 时从 **YOLO polygon txt** 栅格化；`bg_class_idx` 默认 `nc-1`。

#### `__init__`（第 1176–1186 行）

设背景类索引后 `SemanticDataset.__init__`。

#### 方法重绑定（第 1188–1195 行）

label 路径/hook 用 **YOLODataset** 实现，避免 Semantic 的 mask_files 逻辑。

#### `_read_mask`（第 1197–1218 行）

无 segment→全图背景类；否则反归一化多边形，`polygons2masks_overlap` 得实例图；`nc==1` 前景=1，else 按 cls 填类 id。

---

### `ClassificationDataset`（第 1221–1460 行）

**不继承 BaseDataset**；`ImageFolder` + torchvision transforms；样本行 `[path, class_idx, npy_path, im, (tile, source_index)?]`。

#### `__init__`（第 1247–1345 行）

懒 import torchvision；`fraction` 子采样；`cache_ram`/`cache_disk`；`verify_images` 过滤坏图；`check_class_names` 对齐模型类名，**drop 模型没有的文件夹**；`single_cls` 全索引 0；ndjson 文件夹名为类 id；tile 时用 PIL `exif_size`+`tile_windows`；`source_offsets` 仅验证；`classify_augmentations` 或 `classify_transforms(imgsz)`。

#### `__getitem__` / `get_image_and_label`（第 1347–1374 行）

RAM：`img_cache[source]`；disk npy 同 Base 刷新策略；tile crop；BGR→RGB PIL→`torch_transforms`；输出 `img` tensor、`cls` int。

#### `collate_fn`（第 1376–1387 行）

`default_collate` 排除 tile 元数据；有 tile 则 tuple 保留 `tile`/`source_index`。

#### `__len__`（第 1389–1391 行）

`len(samples)`。

#### `imread`（第 1393–1398 行）

包装 `patches.imread`。

#### `cache_images`（第 1400–1416 行）

线程池读全图→ **`BaseDataset._ImageCache`** 单 buffer。

#### `verify_images`（第 1418–1460 行）

尝试加载 `root.cache`（hash=路径+classes）；失败则 `verify_image` 并行扫描，写 cache，返回有效 samples 列表。