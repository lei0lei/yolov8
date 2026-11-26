# `ultralytics/data/utils.py`

本文件是 Ultralytics 数据管线的「工具枢纽」：常量（图像/视频格式、YAML 键类型）、路径映射、深度 PNG/NPY、标签与图像校验（含 pose/segment/semantic/depth）、数据集 YAML 解析与自动下载、分类数据集检查、mask 栅格化、缓存读写、NDJSON 转接等。对外大量符号被 `data/base.py`、`data/dataset.py`、`data/build.py`、`engine/*` 及 `converter.py`、`split*.py` 引用；训练前几乎必经 `check_det_dataset` 或 `check_cls_dataset`。模块不定义 Dataset 类，但承载缓存版本、多任务 verify 回调与 `img2label_paths` 约定。

## 逐段说明

### 导入区（L1–L36）

- **标准库**：`os`、`random`、`subprocess`、`time`、`zipfile`、`Path`、`is_tarfile`、`Any`、`uuid4`。
- **图像/数值**：`cv2`、`numpy`、`PIL.Image`/`ImageOps`。
- **`ultralytics.nn.autobackend.check_class_names`**：规范化类名为 `dict[int,str]`。
- **`ultralytics.utils` 核心**：资源 URL、`DATASETS_DIR`、`LOGGER`、`ROOT`、`SETTINGS_FILE`、`YAML`、URL/颜色工具、`is_dir_writeable`。
- **`utils.checks`**：字体、`check_file`、`is_ascii`、`normalize_platform_uri`。
- **`utils.downloads`**：`download`、`safe_download`。
- **`utils.ops.segments2boxes`**：多边形 → 外接 xywh。
- **`utils.patches.imread`**、**`plotting.Colors`**。

### 模块级常量（L38–L66）

- **`HELP_URL`**：数据集格式文档链接。
- **`IMG_FORMATS` / `VID_FORMATS`**：小写扩展名集合；加载器与校验共用。
- **`FORMATS_HELP_MSG`**：报错时拼接支持列表。
- **`DATASET_KEY_TYPES`**：`check_det_dataset` 对 YAML 键做类型检查的白名单。
- **`DEPTH_PNG_SCALE=1000`**：深度 PNG 默认毫米标度；0 表示无效像素。

### `tile_windows`（L69–L77）

- 输入图像 `(h,w)`、窗 `size`（int 或 `(th,tw)`）、重叠比 `overlap`。
- **y/x 步进**：`range(0, max(dim-size,0), round(size*(1-overlap)))`，并强制包含贴边窗 `(max(h-th,0), max(w-tw,0))`。
- **返回**：`(x1,y1,x2,y2)` 列表，窗不越界。供大图滑窗推理/训练辅助。

### `save_depth_png`（L80–L112）

- **scale 校验**：正有限数，否则 `ValueError`。
- **维度**：`float32`；squeeze 掉为 1 的轴直至 2D。
- **valid**：有限且 `depth>0`；编码 `uint16`，`rint(depth*scale)`，超 `uint16` 上限报错并提示降低 `depth_scale`。
- **无效像素**：编码为 0。
- **`cv2.imwrite` 失败**：`OSError`。

### `load_depth`（L115–L143）

- **`.npy`**：须 2D 浮点；`nan_to_num` 负/NaN→0，单位米。
- **PNG**：scale 校验；PIL 须 PNG 且模式 `I`/`I;16`；值除以 `scale` 得米；返回 `float32`，0=无效。

### `img2label_paths`（L146–L158）

- 将路径中**最后一处** `{sep}images{sep}` 替换为 `{sep}{label_dir}{sep}`（默认 `labels`），扩展名改为 `suffix`（默认 `.txt`）。
- **批量**输入列表，返回平行标签路径字符串列表。全库 YOLO 图-标配对约定。

### `check_file_speeds`（L161–L237）

- **空列表**：warning 返回。
- **抽样**：最多 `max_files` 个随机文件。
- **每文件**：`stat` 耗时作 ping；整文件 `read` 算 MB/s。
- **异常**：单文件失败静默跳过。
- **无 ping 数据**：warning。
- **判定快**：`avg_ping < threshold_ms` 且（`avg_speed > threshold_mb` 或平均文件 `<16KB`）；否则 warning 建议本地盘。
- **调用**：`BaseDataset` 等初始化时抽样提示 IO 瓶颈。

### `get_hash`（L240–L253）

- SHA256：路径字节 + NUL + `size:mtime_ns` + NUL；`stat` 失败则空字段。用于数据集标签缓存失效判断。

### `exif_size`（L256–L265）

- 默认 `img.size` (宽,高)。
- **EXIF 274 旋转** 5–8：JPEG/MPO/PNG/AVIF 交换宽高；否则 TIFF 等用 `imread(filename)` 形状。
- 异常吞掉，返回原尺寸。

### `check_image`（L268–L295）

- `Image.open`+`verify`；`exif_size` 再转 `(h,w)`。
- **断言**：任一边 >9 像素；格式在 `IMG_FORMATS`（外加 jpeg2000）。
- **JPEG 尾**：非 `ffd9` 则 `_replace_image` 用 `_exif_jpeg` 重存 quality=100，msg 说明修复。
- **返回**：`(msg, shape)`。

### `_exif_jpeg`（L298–L301）

- 上下文打开图，`ImageOps.exif_transpose` 后返回已定向副本（文件已关）。

### `_replace_image`（L304–L312）

- 临时文件 `.stem.uuid.suffix`，`write(tmp)` 后 `os.replace` 原子覆盖；`finally` 删临时。避免硬链接共享文件被误改。

### `verify_image`（L315–L335）

- **args**：`((im_file, cls), prefix)`，分类缓存 worker 格式。
- **成功**：`check_image` msg 加前缀，`nf=1`。
- **异常**：`nc=1`，忽略 corrupt 消息。
- **返回**：`((im_file,cls), nf, nc, msg)`。

### `verify_image_depth`（L338–L380）

- **args**：`(im_file, depth_file, prefix, scale)`。
- **`check_image`** 通过后：深度文件缺失 → `nm=1`，返回全 None。
- **`.npy`**：mmap 2D 浮点；否则 PNG 模式断言 + `verify`。
- **纵横比**：`log((dw/dh)/(iw/ih))` ≤0.02，否则 assert。
- **成功**：`nf=1` 与 `im_file, shape`。
- **异常**：`nc=1`，返回 None 占位。

### `read_mask`（L383–L403）

- **`mode=="P"`**：调色板图；若调色板灰度且与 convert L 后颜色数一致则用 `L`，否则用索引图，得类 id `(H,W)`。
- **其它**：`cv2.IMREAD_ANYDEPTH`；失败 `FileNotFoundError`。
- **`mode=="1"`**：值除以 255（1-bit 前景 255→1）。

### `verify_image_mask`（L406–L441）

- **args**：`(im_file, mask_file, prefix, invalid)`，`invalid` 为 256 长度查找表，标记非法 id。
- mask 存在：`read_mask`、形状与图一致、`invalid[mask]` 无真值。
- **缺失 mask**：`nm=1`。
- **返回**：含 `mode` 供后续单次 `read_mask`；失败时前四字段 None。

### `verify_image_label`（L444–L521）

**检测/分割/姿态标签核验核心；>30 行分三段。**

#### 段 1：读图与读标签（L459–476）

- 初始化计数 `nm,nf,ne,nc` 与 `segments,keypoints`。
- `check_image` → shape。
- 标签存在则 `nf=1`，UTF-8 读行 split。
- **pose 配置但非 keypoint 任务**：仅保留 5 列框。
- **分割行**（任一行 len>6 且无 keypoint）：不得混 5 列检测行；类 + 点列 reshape (-1,2)；`segments2boxes` 得 xywh；`lb` 变 `(N,5)`。

#### 段 2：数值与类约束（L477–504）

- 非空：`keypoint` 时列数 `5+nkpt*ndim`；否则 5 列。
- 坐标 ≤1.01；负值规则（宽高可略负容差）；类 id 整数。
- `max_cls < num_cls`（`single_cls` 时 max 视作 0）。
- **去重**：`unique` 行；若有 segments 用 cls+polygon bytes 去重；删减则 warning。

#### 段 3：空标与姿态输出（L505–521）

- 空文件 `ne=1`，零行 placeholder。
- 无文件 `nm=1` 同样零行。
- **keypoint**：reshape `(N,nkpt,ndim)`；`ndim==2` 时拼可见性第三维。
- **返回**：`lb` 仅前 5 列；带 segments/keypoints 与计数。
- **异常**：`nc=1`，前五 None 的 list 形式。

**磁盘标签格式（检测）**：`cls cx cy w h` 归一化。分割文件可同时保留多边形行（verify 会生成 segments 并行 boxes）。

### `visualize_image_annotations`（L524–L565）

- 读 EXIF 校正图；读 txt 每行 5 个 float 检测框（不支持 seg/pose 行）。
- 归一化中心宽高 → 像素 xywh；matplotlib 画框与类名，颜色 `colors(label)`，文字对比度按亮度。
- `plt.show()` 交互显示；非训练路径。

### `polygon2mask`（L568–L590）

- 零 mask `uint8`；点多边形 `fillPoly`；再 `resize` 到 `imgsz/downsample_ratio`。注释：先 fill 再 resize 与训练 mask 损失一致。

### `polygons2masks`（L593–L609）

- 列表推导：每个 polygon 单独 `polygon2mask`，堆成 `(N,H,W)`。

### `polygons2masks_overlap`（L612–648）

- 每 segment 单独 mask、求 area；按面积降序索引。
- **叠加**：`np.maximum(masks, ms[i]*(i+1))` 避免 uint8 溢出（旧加法 Bug）。
- **返回**：像素值 `i+1` 表实例 id（小实例在上）+ 排序索引。

### `find_dataset_yaml`（L651–L669）

- 先 `glob` 根 `*.yaml`，无则 `rglob`；无则 assert。
- 多个时优先 `stem == path.stem`；须唯一。

### `get_split_fraction`（L672–L697）

- 拒绝 bool。
- **list fraction**：按 `train/val/test` 取下标，缺省 1.0。
- 非 train 且标量 fraction → 1.0。
- `0/1` 边界转 float。
- train/val 的 0 → `ValueError`。

### `convert_ndjson_to_yolo_if_needed`（L700–L722）

- `normalize_platform_uri`。
- 若 URL/路径为 `.ndjson` 或 `ul://.../datasets/`：`asyncio.run(convert_ndjson_to_yolo)`。
- 否则原样返回 `data`。

### `check_det_dataset`（L725–843）

**训练 val/predict 前数据集解析；分三段。**

#### 段 1：定位文件与解压（L740–756）

- 非 URL 且路径不存在且非 yaml 后缀：尝试补 `.yaml`/`.yml`。
- `check_file`；目录则 `find_dataset_yaml`。
- zip/tar：`safe_download` 解压到 `DATASETS_DIR`，`autodownload=False`，`extract_dir` 为 yaml 父目录。

#### 段 2：YAML 语义校验（L756–796）

- `DATASET_KEY_TYPES` 类型检查。
- 必须有 `train`/`val`（`validation` 重命名为 `val`）。
- `split` 指定但无路径 → `FileNotFoundError`。
- `names` 或 `nc` 必填；`nc` 整型化；`names` 长度与 `nc` 一致；缺 names 则自动生成 `class_i`。
- `check_class_names`；`colors` 解析失败则删除键；`channels` 默认 3。

#### 段 3：路径解析与下载（L798–843）

- **root**：`extract_dir` 或 `path` 或 yaml 父目录；相对则挂 `DATASETS_DIR`。
- **train/val/test/minival**：字符串或列表均 resolve 绝对路径；`../` 回退尝试。
- **val 不存在**：若有 `download` 且 `autodownload`，执行 HTTP 压缩包或 `exec(s)` 脚本；否则 `FileNotFoundError` 提示 `SETTINGS_FILE`。
- 默认 `masks_dir` 若存在 `masks/` 目录。
- `check_font` ASCII/Unicode。
- **返回**：富 dict，供 `YOLODataset` 等。

**调用**：`YOLO.train/val/predict` 配置 `data=` 时。

### `check_cls_dataset`（L846–953）

#### 段 1：参数与获取 data_dir（L868–898）

- **L868–869**：`split` 非空且不是 train/val/test → `ValueError`。
- **L872–873**：`http/https` 开头 → `safe_download` 到 `DATASETS_DIR` 并解压。
- **L874–878**：本地路径像压缩包后缀 → `check_file` 再 `safe_download` 解压。
- **L880–881**：`dataset` 转 Path；`data_dir` 为已有目录或 `DATASETS_DIR/dataset` 的 resolve。
- **L882–887**：仍非目录：若带扩展名则报错必须是目录；若 `dataset` 含路径分隔符则 `FileNotFoundError`（不给裸名以外的自动下载）。
- **L888–897**：仅官方裸数据集名：imagenet 跑 `get_imagenet.sh`，其它从 `ASSETS_URL/{name}.zip` 下载；成功后继续。

#### 段 2：train/val/test 目录解析（L898–928）

- **L898–908**：`train_set = data_dir/train`；不存在则在 `data_dir` 全树搜 `IMG_FORMATS`，找到则 `split_classify_dataset(..., 0.8)` 并更新 `data_dir` 与 `train_set`；否则 `FileNotFoundError`。
- **L909–917**：`val_set` 依次尝试 `val`、`validation`、`valid`；`test_set` 仅当 `test` 存在。
- **L919–922**：用户要 val/test 但两者皆无 → `FileNotFoundError`。
- **L923–928**：只要 val 却要 val 且无 → 用 test 顶替 warning；只要 test 却无 → 用 val 顶替。

#### 段 3：类名、统计与返回（L930–953）

- **L930–931**：若存在 `.ndjson.yaml`（NDJSON 分类转换产物）则读其中 `names`。
- **L932–934**：否则 `train` 下一级子目录名排序 → `dict(enumerate(...))` 为类名；`nc=len(names)`。
- **L937–951**：对 train/val/test：无目录只打 info；有则 `rglob` 图像数 `nf`、首段子目录数 `nd`。
  - train 的 `nf==0` → `FileNotFoundError`；其它 split 仅 warning。
  - 有图则 info 打 `nf` 与 `nd/nc`。
- **L953**：返回 Path 三件套与 `nc`、`names`、`channels=3`，供 `ClassificationDataset`。

### `compress_one_image`（L956–991）

- **PIL 路径**：解除像素上限；EXIF 定向；RGBA→RGB；超 `max_dim` 等比缩小；JPEG optimize 原子写。
- **失败**：OpenCV 读图、resize、`imwrite` 同样原子替换。

### `load_dataset_cache_file`（L994–L1023）

- 自定义 `Unpickler` 白名单：numpy ndarray/dtype、pathlib Path。
- **读**：`gc.disable`；判断是否 numpy `.npy` 头；`Unpickler.load().item()` 得 dict。
- **finally** `gc.enable`。

### `save_dataset_cache_file`（L1026–L1047）

- 写入 `version` 键；父目录可写则删旧文件 `np.save`；失败删残留 warning；不可写 warning。

### `add_polygon_background`（L1050–L1078）

- 已有 `masks_dir` 或 `_polygon_bg_added` 直接返回。
- **nc==1**：二值语义，`bg_class_idx=0`，不增类。
- **nc>1**：追加 `background` 类 id=nc，`nc+=1`。
- 标记 `_polygon_bg_added` 防重复。

**调用**：多边形语义训练配置，无 PNG mask 目录时。

## 校验与缓存数据流（简图）

```
图像列表 → verify_image_* (多进程) → nf/nm/nc 统计 + 规范化 lb/shape
         ↓
get_hash(labels+images) → load/save_dataset_cache_file
```

## 主要调用方

| 符号 | 典型调用方 |
|------|------------|
| `check_det_dataset` | `engine/trainer.py`, `engine/validator.py`, CLI |
| `check_cls_dataset` | 分类 Trainer/Validator |
| `verify_image_label` 等 | `data/dataset.py` 缓存构建 |
| `img2label_paths` | `dataset.py`, `split.py`, `split_dota.py` |
| `IMG_FORMATS` | 全局数据加载 |
| `convert_ndjson_to_yolo_if_needed` | `check_det_dataset` 前后、训练入口 |
| `add_polygon_background` | 语义分割 polygon 数据集构建 |