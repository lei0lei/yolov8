# `ultralytics/data/base.py`

本文件实现 Ultralytics 视觉训练/验证共用的 **`BaseDataset`**（`torch.utils.data.Dataset` 子类）及 RAM 侧 **`_ImageCache`** 打包机制。职责：解析图像路径列表、加载/缩放图像、可选 **tile 切片**、**rect 矩形 batch** 排序、**RAM/disk 缓存**（`*.npy`）、**mosaic 缓冲**、子类可覆写的 `get_labels` / `build_transforms`。对外主要被 `dataset.py` 中各任务 Dataset 继承；`build_yolo_dataset` 最终实例化子类时传入 `imgsz`、`cache`、`tile` 等参数。单张样本经 `__getitem__` → `get_image_and_label` → `transforms` 输出 dict（含 `img`、几何元数据、子类填充的标注字段）。

**标签 dict 约定（子类 `get_labels` 产出）**：`im_file`、`shape` 为 `(H,W)`；检测类还带 `cls` `(n,1)`、`bboxes` `(n,4)` **xywh 归一化**、`segments`、`keypoints`、`normalized=True`、`bbox_format="xywh"`。Tile 时额外 `source_index`、`source_shape`、`tile` 为 `(x1,y1,x2,y2)`，逻辑 `shape` 为切片高宽。

## 逐段说明

### 导入（第 3–30 行）

- **标准库**：`glob`/`Path` 扫图；`ThreadPool` 并行缓存；`copy`/`deepcopy` 超参与标签拷贝。
- **数值与 DL**：`cv2`、`numpy`、`torch`、`Dataset`。
- **`data.utils`**：`IMG_FORMATS`、`.npy` 原子写 `_replace_image`、`check_file_speeds`、`get_split_fraction`、`tile_windows`。
- **`ultralytics.utils`**：`DEFAULT_CFG`、`LOCAL_RANK`（分布式仅 rank0 进度条）、`LOGGER`、`NUM_THREADS`、`TQDM`、`IterableSimpleNamespace`。
- **`patches.imread`**：统一解码（含 PIL 回退格式）。

### 类 `BaseDataset._ImageCache`（第 82–115 行）

将多张 `uint8` 图像压入**一块连续 `torch.uint8` 张量**，子进程 fork 时共享只读视图；pickle 到 worker 时在 `/dev/shm` 不足时退回整 buffer 拷贝。

#### `__init__`（第 85–94 行）

- 记录各图 `shapes`、`dtypes.str`、`offsets`（前缀和字节界）。
- `torch.empty(offsets[-1])` 分配总字节；用 `buffer.numpy()` 切片把每张图 `reshape(-1).view(uint8)` 写入，原 list 中引用置 `None` 释 Python 引用。

#### `__getitem__`（第 96–101 行）

按索引从连续字节 **view 回原 dtype 并 reshape 为 `shapes[i]`**，返回 `np.ndarray` 视图。

#### `__getstate__` / `__setstate__`（第 103–115 行）

Pickle 时若 tensor 未 `share_memory_` 且 `/dev/shm` 空闲小于 **2× 缓存字节**，警告并把 `buffer` 存为 numpy 数组；反序列化再 `torch.as_tensor` 恢复。

---

### 类 `BaseDataset`（第 117–589 行）

#### `__init__`（第 117–220 行）

**参数落地**：`imgsz` 可为 int（方形边长）或 `(H,W)` 元组；`channels` 决定 `cv2_flag`（灰度/彩色/UNCHANGED）。

**流程分段**：

1. **`get_img_files`**：得到 `im_files`；**`get_labels`**：子类实现；**`update_labels(classes)`**：过滤类别 / `single_cls` 全置 0。
2. **Tile 分支**（`if tile`）：对每个源 label 用 `tile_windows(label["shape"], 窗口, hyp.tile_overlap)` 展开多条 label，`im_files` 改为每条 tile 对应同一源路径；`tile_counts` 记录每源 tile 数。
3. **验证且 tile**（`if not augment`）：`source_offsets = cumsum([0,*tile_counts])`，供分布式 **按源对齐** 切分。
4. **`ni = len(labels)`**；`rect = rect and not (tile and not augment)` — 验证 tile 时保持同源连续，不做 rect 重排。
5. **`set_rectangle()`** 当 `rect` 为真。
6. **Mosaic buffer**：`max_buffer_length = min(ni, batch*8, 1000)` 仅训练；用于非 RAM 缓存时保留最近加载索引。
7. **缓存**：`ims/im_hw0/im_hw` 长度 `ni` 的占位 list；`npy_files = 每图 .npy` 路径（命名约定不可改，否则用户磁盘缓存失效）。`cache` 规范为 `None`/`ram`/`disk`；RAM 先 `check_cache_ram()`，确定性训练时 `hyp.deterministic` 会警告 RAM 非确定性；disk 则 `check_cache_disk()` 后 `cache_images()`。
8. **`build_transforms(copy(hyp))`**：子类构建增广链，不污染调用方 `hyp`。

#### `get_img_files`（第 222–259 行）

- **`try` 内**：路径可为 list；目录则 `glob **/*.*`；单文件扩展名在 `IMG_FORMATS` 则直接加入；否则当 **文本清单** UTF-8 逐行，`./` 相对清单目录展开。
- 过滤扩展名、排序、`assert` 非空；**`fraction`**：若为 int 则取前 N 张，否则按比例 `round`；最后 **`check_file_speeds`** 抽检读速。

#### `update_labels`（第 261–282 行）

遍历 `labels`：若 `include_class`，用广播比较 `cls` 与 `include_class_array` 得掩码 `j`，同步裁剪 `bboxes`/`segments`/`keypoints`；若 `single_cls`，`cls[:] = 0`。

#### `_load_source`（第 284–315 行）

Tile+`source_images` RAM 缓存时直接按 `source_index` 取；否则单 worker 内 `_source_image` 缓存上一张同源全图。

- **disk / 新鲜 `.npy`**：`np.load`；通道数与 `self.channels` 不符或 `uint16` 则刷新：警告、`imread` 原图、`_replace_image` 写回 npy。
- 损坏 npy：unlink 后 `imread`。
- 无缓存：`imread(f, cv2_flag)` BGR。
- `im is None` → `FileNotFoundError`；tile 时更新 `_source_image`。

#### `load_image`（第 317–376 行）

**已 RAM 命中**（`self.ims[i]` 非空）：直接返回缓存的三元组。

**未命中**：

1. `_load_source(i)`；tile 则按 `labels[i]["tile"]` **数组切片 `.copy()`**。
2. `h0,w0 = shape[:2]`；`shape` 目标：`imgsz` 元组或 `(imgsz,imgsz)`，`imgsz = max(shape)`。
3. **`rect_mode`**：`resize_short` 时短边缩放到 `imgsz`（`r = imgsz/min(h0,w0)`，ceil 另一边）；否则长边缩放到不超过 `imgsz`（`r = imgsz/max(h0,w0)`，`w,h` 均 `min(ceil*r, imgsz)`）。`r!=1` 时 `cv2.resize` LINEAR。
4. 非 rect 且尺寸不等：**拉伸**到 `shape[::-1]`（cv2 要 W×H）。
5. 灰度 2D → `[..., None]` 增通道维。
6. **训练且非 RAM 全缓存**：写入 `ims[i]` 与 hw 元数据，push `buffer`；超长则 pop 最旧索引并清空其 RAM 槽（释放 mosaic 外图像）。

返回 `(im, (h0,w0), im.shape[:2])` 分别为 resize 后图、原图尺寸、resize 后尺寸。

#### `cache_images`（第 378–406 行）

**Tile + RAM**：`indices` 为各 `source_index` 代表索引；预加载到 `images` list，再 `_ImageCache` → `source_images`，清空 `_source_image`。

**否则**：`disk` 用 `cache_images_to_disk` 线程池 `imap`；`ram` 用 `load_image`（会填 `ims`）。进度条累计 GB；**RAM 结束**再 `self.ims = _ImageCache(self.ims)` 合并。

#### `cache_images_to_disk`（第 408–418 行）

`npy` 不存在或 mtime 早于原图：`_replace_image` 原子写 `imread` 结果；异常则 unlink 并警告。

#### `check_cache_disk`（第 420–452 行）

随机最多 30 张估 `nbytes`，外推 `disk_required = b * len(files)/n * (1+safety_margin)`；父目录不可写则 `cache=None`；`shutil.disk_usage` 空闲不足则禁用缓存并警告。

#### `check_cache_ram`（第 454–483 行）

Tile 时对 **源图** `_load_source` 估字节，否则 `load_image[0]`；`mem_required` 与 `psutil.virtual_memory().available` 比较，`safety_margin` 默认 1.0（100% 余量）。

#### `set_rectangle`（第 485–508 行）

- `bi = floor(arange(ni)/batch_size)` 为每张图的 batch 内序号；`nb` 为总 batch 数。
- 从各 label **`pop("shape")`** 得 `(h,w)`，算宽高比 `ar`，`argsort` 重排 `im_files`/`labels`。
- 每个 batch 内：若 `maxi<1` 偏竖图，batch 形状 `[maxi,1]`；若 `mini>1` 偏横图，`[1,1/mini]`。
- **`batch_shapes`** = `ceil(shapes * imgsz/stride + pad) * stride`，整除 stride 的 **letterbox 目标 (h,w) 模板**；`self.batch = bi`。

#### `__getitem__`（第 510–512 行）

`transforms(get_image_and_label(index))` — 子类 Format 在此之后把 numpy 转 tensor。

#### `get_image_and_label`（第 514–542 行）

`deepcopy(label)`，去掉缓存用 `shape` 键。

- **`full=True`**：去 tile 元数据，`_load_source` 全图；`ori_shape`/`resized_shape` 均为原图 hw；`ratio_pad` 二级为 `(ratio, (0,0))`。
- **否则**：`load_image(index)` 得三分量。
- **`ratio_pad`**：resize 后与原图高宽比 `(rh, rw)`。
- **rect**：附加 `rect_shape = batch_shapes[batch[index]]`。
- 最后 **`update_labels_info`**（子类打包 `Instances` 等）。

#### `__len__`（第 544–546 行）

`len(self.labels)` — tile 后为 tile 条数。

#### `update_labels_info` / `build_transforms` / `get_labels`（第 548–589 行）

基类默认透传或 `NotImplementedError`；子类必须实现后两者。