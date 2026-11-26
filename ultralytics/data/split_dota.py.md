# `ultralytics/data/split_dota.py`

本模块将 DOTA 大图（尤其是遥感场景）按滑动窗口裁成小图，并按窗口重算 YOLO OBB 标签（归一化四顶点 + 类 id）。对外公开函数：`bbox_iof`、`load_yolo_dota`、`get_windows`、`get_window_obj`、`crop_and_save`、`split_images_and_labels`、`split_trainval`、`split_test`。典型用于 DOTA 训练前预处理脚本或 `if __name__` 示例；标签加载路径与 `converter.convert_dota_to_yolo_obb` 产出的 `labels/{train|val}/*.txt` 格式一致。依赖 `shapely` 计算多边形与矩形窗的精确交集（IoF）。

## 逐段说明

### 导入区（L1–L18）

- **标准库**：`itertools.product` 生成窗口左上角网格；`glob` 列举图像；`ceil` 计算滑动步数；`Path`、`typing.Any`。
- **第三方**：`cv2` 写裁切图；`numpy` 数组运算。
- **`ultralytics.data.utils`**：`IMG_FORMATS`、`img2label_paths`。
- **`ultralytics.utils.instance.Instances`**：裁切后裁剪、归一化 OBB 顶点。
- **`imread`**：统一读图（含 EXIF 等补丁行为）。
- **`check_requirements("shapely>=2.0.0")`**：在 `bbox_iof` 内延迟校验。

### `bbox_iof`（L21–L64）

**职责**：计算 N 个 OBB 多边形（每对象 4 点）与 M 个轴对齐窗口矩形之间的 **Intersection over Foreground**（交集面积 / 多边形面积）。

#### 粗筛与矩形化（L36–L47）

- **L36–L37**：按需安装并导入 `shapely.geometry.Polygon`。
- **L39**：`polygon1`  reshape 为 `(N, 4, 2)`。
- **L40–L42**：对每个多边形取 min/max 得外接 AABB `bbox1`，形状 `(N, 4)`，用于与 `bbox2` 快速求交。
- **L44–L47**：标准轴对齐框 IoU 式重叠面积 `h_overlaps`，形状 `(N, M)`；无重叠的对跳过精确多边形求交。

#### 精确交集（L49–L64）

- **L49–L50**：将每个 `bbox2` 转为四顶点矩形多边形 `polygon2`。
- **L52–L56**：仅对 `h_overlaps > 0` 的 `(i,j)` 用 Shapely 算 `intersection.area`。
- **L57–L58**：分母为 `polygon1` 面积（前景），非并集。
- **L60–L63**：`eps` 防除零；一维时扩维保持 `(N, M)`。

**格式**：`polygon1` 为 `(N, 8)` 扁平坐标 `[x1,y1,...,x4,y4]`；`bbox2` 为 `(M, 4)` `[xmin, ymin, xmax, ymax]`（像素）。

### `load_yolo_dota`（L67–L99）

- **L87–L88**：`split` 仅 `train`/`val`；图像目录必须存在。
- **L90**：`glob` 取目录下文件，扩展名过滤 `IMG_FORMATS`。
- **L91**：`lb_files = img2label_paths(im_files)` — 将路径中 `/images/` 换为 `/labels/` 并改后缀 `.txt`。
- **L93–L98**：逐对读取；无标签文件则 `lb=[]`（背景图）。
- **标签解析**：每行空格分割，转 `float32` 数组；期望行宽 9：`cls + 8 个归一化 OBB 坐标`（与 YOLO OBB 训练一致）。
- **返回**：`list[dict]`，键 `label`、`filepath`。

### `get_windows`（L102–L151）

**职责**：多组 `(crop_size, gap)` 生成滑动窗口坐标，并丢弃「窗内有效图像面积占比」过低的窗。

#### 网格生成（L121–L139）

- **L121–L125**：`h,w` 为原图尺寸；`crop_size > gap`，步长 `step = crop_size - gap`。
- **L127–L130**：列方向窗口数 `xn`；`xs` 为各窗左上角 x；若最后一窗越界，对齐到 `w - crop_size`。
- **L132–L135**：行方向同理。
- **L137–L139**：`product(xs, ys)` 得起点，`stop = start + crop_size`，拼接为 `(N, 4)` `[x0,y0,x1,y1]`。

#### 过滤（L142–L151）

- **L142–L144**：将窗坐标 clip 到图像边界，算「窗内有效区域」面积与理论窗面积比 `im_rates`。
- **L148–L150**：若无窗超过 `im_rate_thr`，把最接近最大率的窗强制保留（`eps` 容差），避免空结果。
- **返回**：满足 `im_rates > im_rate_thr` 的 `windows` 行。

### `get_window_obj`（L154–175）

- **L166–L167**：从 `anno` 取 `ori_size` 与 `label`。
- **L168–L170**：若有标签，将归一化 OBB **原地**乘以 `w/h` 变为像素坐标（调用方需注意 `anno` 被修改）。
- **L171**：`bbox_iof(label[:, 1:], windows)` — 用 8 坐标多边形对外接过程已在 `bbox_iof` 内处理… 实际传入的是 `label[:, 1:]` 即 8 维 OBB，reshape 在 bbox_iof 内完成。
- **L173**：对每个窗 `i`，取 `iofs[:, i] >= iof_thr` 的行，得到该窗内实例标签（仍为像素坐标，含 cls 列）。
- **L174–L175**：无标签时返回 `M` 个空 `(0, 9)` 数组。

### `crop_and_save`（L178–228）

**单张大图 → 多窗小图 + 标签**。

#### 读图与分窗（L206–L209）

- **L206**：`imread` 原图；`ori_size = (H, W)`。
- **L208–L209**：`get_windows` + `get_window_obj` 得每窗对象集合。

#### 逐窗处理（L211–L228）

- **L212–L213**：窗像素边界；输出 stem 命名 `{name}__{宽}__{x}___{y}`，避免不同裁切冲突。
- **L214–L215**：numpy 切片得 patch；`ph, pw` 为 patch 尺寸。
- **L217–L219**：有标签或允许无标签时，写 `patch_im` 为 `.jpg`。
- **L220–L228**：有标签时：
  - **L221**：`Instances` 用零占位框 + 顶点减 `(x_start, y_start)` 转到窗坐标系。
  - **L222**：`clip(pw, ph, preserve_obb=True)` — IoF 入选的对象可能超出窗界，裁到窗内。
  - **L223**：顶点再除以 `(pw, ph)` 归一化到 `[0,1]`，写回 `label[:, 1:]`。
  - **L225–L228**：每行 `int(cls)` + 8 个 `%6g` 坐标，YOLO OBB txt 格式。

**`lb_dir` 为空**（`split_test`）：只写图不写标签。

### `split_images_and_labels`（L231–268）

- **L261–L264**：在 `save_dir/images|labels/{split}` 建目录。
- **L266–L268**：`load_yolo_dota` 后 tqdm 逐条 `crop_and_save`。

### `split_trainval`（L271–306）

- **L301–L304**：对每个 `rates` 中比例 `r`，`crop_size/r`、`gap/r` 取整，得到多尺度窗（大图用小窗、等价更密采样）。
- **L305–L306**：对 `train`、`val` 各调 `split_images_and_labels`。

### `split_test`（L309–342）

- **L331–L334**：同样按 `rates` 算 `crop_sizes`、`gaps`。
- **L335–L336**：仅 `save_dir/images/test`。
- **L338–L340**：列举 `data_root/images/test` 图像。
- **L341–L342**：`anno` 无标签 `(0, 9)`，`lb_dir=""`，只裁图。

### `__main__`（L345–347）

- 示例：`split_trainval` + `split_test`，`data_root="DOTAv2"`，`save_dir="DOTAv2-split"`。

## 坐标与格式转换小结

| 阶段 | 表示 |
|------|------|
| 磁盘标签 | 每行：`cls x1 y1 … x4 y4`，归一化相对原图 |
| `get_window_obj` 内 | 临时转为原图像素坐标 |
| 裁切后写盘 | 相对 **patch** 宽高的归一化 OBB |
| 窗与对象匹配 | IoF ≥ 阈值（默认 0.7） |