# `ultralytics/data/converter.py`

本模块汇集多种「外部标注 / 平台格式 → Ultralytics YOLO 训练结构」的转换器：COCO JSON、灰度语义 mask、DOTA OBB、检测框转 SAM 分割、合成 COCO、RGB→伪多光谱 TIFF、以及异步 NDJSON（含 Platform）全量落地。对外主要为各 `convert_*` 顶层函数与 COCO 类 ID 映射表；`utils.convert_ndjson_to_yolo_if_needed` 与 `check_det_dataset` 链路会调用 `convert_ndjson_to_yolo`。模块复用 `utils.get_split_fraction`、`read_mask`，并大量使用 `increment_path`、`safe_download`、`TQDM` 等工具。

## 逐段说明

### 导入区（L1–L37）

- **标准库**：`asyncio`（NDJSON）、`hashlib`、`json`、`math`、`os`、`random`、`shutil`、集合 `defaultdict`、线程池、`Path`、`TemporaryDirectory`、`uuid4`。
- **图像**：`cv2`、`numpy`、`PIL.Image`。
- **`ultralytics.data.utils`**：`get_split_fraction`、`read_mask`。
- **`ultralytics.utils` 族**：数据集目录、`LOGGER`、`NUM_THREADS`、`PLATFORM_URL`、进度条、`WINDOWS`、`YAML`、`clean_url`；`check_file`、`check_requirements`；`download`、`zip_directory`；`increment_path`；`imread`。

### `coco91_to_coco80_class`（L40–L139）

- 返回长度 91 的列表：下标为 COCO 91 类 id（1-based 减 1 后作索引），值为 80 类训练 id 或 `None`（无对应类）。
- 无分支；纯静态映射表，供 `convert_coco` 在 `cls91to80=True` 时使用。

### `coco80_to_coco91_class`（L142–L243）

- 返回长度 80 的列表：每个 80-index 类对应 paper 91-index id。
- 与上函数互为参考，本文件内 `convert_coco` 未直接调用。

### `convert_coco`（L246–390）

**输入**：`labels_dir` 下 `*.json`（COCO instances）；**输出**：`save_dir/images`、`save_dir/labels` 下按相对路径组织的 `.txt`。

#### 初始化（L280–L286）

- **`increment_path(save_dir)`**：目录已存在则自动后缀递增，避免覆盖。
- **L282–L283**：创建 `labels`、`images` 子目录。
- **L286**：加载 91→80 映射表。

#### 逐 JSON（L289–388）

- **L290**：非 LVIS 时去掉 `instances_` 前缀作 labels 子文件夹名。
- **L292–L293**：读入 COCO dict。
- **L296**：`images` 以 `id` 为键的字典。
- **L298–L300**：`annotations` 按 `image_id` 分组。

**RLE 分支（L301–L305）**

- 若 `use_segments` 且存在非 crowd 的 dict 型 `segmentation`，要求 `faster-coco-eval`，导入 `decode`、`segmToRle`。

**逐图写标签（L307–L374）**

- **L310–L312**：图高宽；路径来自 `coco_url`（LVIS）或 `file_name`。
- **L314–L316**：累计 bboxes、segments、keypoints 列表。
- **逐 ann（L317–L367）**：
  - **crowd** 跳过。
  - **bbox**：COCO 为 `[x_tl, y_tl, w, h]` → 加半宽高变中心 → 除以 `w/h` 归一化 → 无效宽高跳过。
  - **cls**：`category_id-1` 再经 `coco80[...]` 或直减 1。
  - **keypoints 模式**：无可见关键点（`[2::3]` 全 0）跳过；否则 bbox 字段与 `keypoints/(w,h,1)` 拼接为一行。
  - **segments 模式**：
    - dict RLE 且 `size==[h,w]`：decode → 外轮廓 → 多边形列表。
    - 过滤合法多边形（≥6 坐标、偶数个、全数字）。
    - 无多边形：`dropped=True`，用 bbox 生成矩形四点伪多边形。
    - 多段：`merge_multi_segment` 后归一化拼接。
    - 单段：坐标展平后 `/ [w,h]` 归一化。

**写文件（L369–L374）**

- **label_file**：相对 `save_dir/labels/...`，后缀 `.txt`；`mkdir` 支持 `file_name` 含子目录。
- **打开模式**：LVIS 用 `"a"`；标准 COCO 用 `"x"` 防止不同扩展名图共享同一 label 文件。
- **行内容**：keypoints > segments > bboxes；`dict.fromkeys(map(tuple, rows))` 去重行。

**收尾（L376–388）**

- 若有 segment 回退为框且无 keypoints，打 warning。
- LVIS：额外写列表 txt，每行 `./images/...` 相对 coco_url。
- 完成日志。

**YOLO 行格式（检测）**：`cls cx cy w h`（归一化）。分割：`cls x1 y1 x2 y2 ...`。姿态：`cls cx cy w h` + 归一化关键点序列。

### `convert_segment_masks_to_yolo_seg`（L393–476）

- **L431–L435**：mask 文件 stem 大小写不重复，否则 `FileExistsError`。
- **映射**：像素值 `i+1` → 类 `i`，0 为背景。
- **每 mask**：
  - `read_mask` 得 `(H,W)` 类 id 图。
  - 对每个非 0 像素值，二值化 → `findContours` → 每轮廓 ≥3 点则一行：`class_index` + 归一化轮廓点（6 位小数）。
- **输出**：`output_dir/{stem}.txt`，多实例多行。

### `convert_dota_to_yolo_obb`（L479–568）

- **L514–L533**：DOTA 类名 → 0–17 索引字典。
- **内部 `convert_label`（L535–553）**：
  - 读 `labels/{phase}_original/{name}.txt`；写 `labels/{phase}/{name}.txt`。
  - 行字段：前 8 个为像素四顶点，第 9 个为类名；不足 9 字段跳过。
  - 坐标偶数位除 `image_width`、奇数位除 `image_height` 归一化。
  - 输出：`class_idx` + 8 个 `%6g` 坐标。
- **主循环（L555–568）**：仅 `.png` 且存在 original 标签；`cv2.imread` 取 `h,w` 再转换。无 original 标签的 PNG 跳过（背景图无输出 txt）。

### `min_index`（L571–583）

- 两 `(N,2)` 与 `(M,2)` 点集，广播求平方距离和，`argmin` 返回索引对。用于 `merge_multi_segment` 找相邻段最近点。

### `merge_multi_segment`（L586–634）

- 各段 reshape 为 `(-1,2)`；相邻段用 `min_index` 记连接点。
- **两轮 `k in range(2)`**：
  - **k=0 正向**：按索引 `roll`、闭合点、中间段截取子弧；首尾段整段保留。
  - **k=1 反向**：补连中间段剩余部分。
- **返回**：连接后的 ndarray 列表，供 COCO 多部件目标合成单条多边形。

### `yolo_bbox2segment`（L637–698）

- 延迟导入 `SAM`、`YOLODataset`、`xywh2xyxy`。
- **`YOLODataset(im_dir, data={"names": range(1000), ...})`**：仅用其标签扫描；若任一 label 已有 `segments` 则直接返回。
- **SAM 循环**：空 label 跳过；框从归一化 xywh 乘 `w,h` 变像素；`xywh2xyxy` 喂 SAM；用 `masks.xyn` 与过滤后的 `cls` 写回内存 label。
- **写出**：默认 `im_dir` 同级 `labels-segment/`，保持相对子目录；每行 `cls` + 归一化多边形坐标，`%g` 格式。

**调用**：用户脚本或与 `annotator` 类似的半自动标注。

### `create_synthetic_coco_dataset`（L701–753）

- 下载 `coco2017labels-segments.zip` 到 `DATASETS_DIR/coco`。
- 删 `labels/test2017`。
- 线程池：读 `train2017.txt`/`val2017.txt` 中路径，若图不存在则随机 RGB 尺寸 480–640 的 PIL 图保存。
- 用于无真实图像仅有标签列表的测试集构造。

### `convert_to_multispectral`（L756–810）

**目录模式（L779–795）**

- 递归所有非 tif 的 `IMG_FORMATS` 图；同 stem 输出 `.tiff` 不可重复。
- 递归调用单图逻辑；`replace` 删原图；`zip` 则 `zip_directory(path)`。

**单图（L797–809）**

- `imread` BGR；波长锚点 `xp=[475,510,650]` nm；`n_channels` 个目标波长线性插值（可外推）得 `multispectral`。
- 多页 TIFF：`save_all`、LZW、`tiffinfo 317:2` 水平预测，与 OpenCV TIFF 习惯对齐。

### `_infer_ndjson_kpt_shape`（L813–857）

- 扫描最多 50 条 pose 标注；关键点长度为 `len(ann)-5`。
- 长度必须一致，否则 `ValueError`。
- 若 `n%3==0` 且可见性均在 `{0,1,2}` → `[n//3, 3]`。
- 否则若 `n%2==0` 且 `n%3!=0` → `[n//2, 2]`。
- 否则报错提示配置 `kpt_shape`。

### `convert_ndjson_to_yolo`（L860–948）

**公共异步入口**：文件锁防并发重复转换；缓存 `.stem-hash.cache` 存结果相对路径。

#### 参数与缓存键（L905–L914）

- `output_path` 默认 `DATASETS_DIR`。
- `fraction` 经 `get_split_fraction` 归一化；`source_hash` 含 source、fraction、split。

#### 锁与转换（L917–948）

- **`convert()`**：删 cache 标记；临时目录 `check_file` 下载源；调 `_convert_ndjson_to_yolo`；写 cache 文本。
- **文件锁**：Windows `msvcrt` / POSIX `fcntl` 非阻塞；等待时若 cache 已有效则直接返回。
- 否则 `await convert()`。

### `_convert_ndjson_to_yolo`（L951–1258）

**体量最大；分三段说明。**

#### 段 1：解析元数据与校验（L959–988）

- 需要 `aiohttp`。
- 线程池读 NDJSON 全线；首行 `dataset_record`，余下 `image_records`。
- `task`：`classify` / `depth` / 其他；depth 校验正有限 `depth_scale`。
- 分类：仅保留有 `annotations.classification` 的记录；ID 必须为 int。
- 本地非分类非 depth 可 `pop("path")` 作图像相对根。

#### 段 2：split 分数、哈希、目录布局（L992–1119）

- **train/val 专用 fraction**：`split==train` 或特定 val 条件时把 test 分数置 0。
- **逐记录改 `file` 名**：防碰撞、保留内容 hash stem；校验 split、file 名；本地 path 拼 `images/{split}`。
- **内容哈希 `_hash`**：决定输出目录 `{stem}-{_hash}`；`metadata` 若 `hash`+`complete` 匹配则短路返回。
- **无 val**：从 train 随机 10%（至少 1 张）划 val，`Random(0)`。
- **类名 / nc**：从标注推断；pose 无 `kpt_shape` 则 `_infer_ndjson_kpt_shape`。
- **`selected`**：按 split fraction 用 `np.linspace` 均匀抽样子集。
- **非分类**：建 `images/{split}`、`labels` 或 `depth/{split}`；组装 `data_yaml`。

#### 段 3：下载、写标签、落盘（L1124–1258）

- **Platform 池**：`pooled_path` 对 32 hex stem URL 硬链接到 `.ndjson-assets`。
- **`ensure_file`**：已存在 / 本地 Path copy / 池 link / HTTP 三次重试；原子 `tmp`+`replace`；成功则 link 入池。
- **`process_record`**：分类 → `{split}/{class_index}/image`；否则 `images`+`labels`（首个 annotation 键的行写 txt）；depth 再下 `depth/*.png`。
- **并发**：`Semaphore(min(128,n))`；每 1024 条 `gather`。
- **失败**：成功数不足则 `RuntimeError`。
- **分类**：写 `.ndjson.yaml`（names、hash、complete）返回目录；否则 `YAML.save data.yaml` 返回 yaml 路径。

**NDJSON → YOLO 结构**

| task | 目录 | 元数据 |
|------|------|--------|
| detect/seg/pose/obb | images+labels/{split} | data.yaml |
| depth | images+depth/{split} | data.yaml + depth_scale |
| classify | {split}/{六位类id}/ | .ndjson.yaml names |

**调用方**：`utils.convert_ndjson_to_yolo_if_needed` → `asyncio.run`；训练 `data=*.ndjson` 或 `ul://` URI。