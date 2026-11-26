# `ultralytics/data/loaders.py`

本模块为 **`yolo predict`** 提供可迭代的**推理数据源**（非 `torch.utils.data.Dataset` 训练集）：网络流、截图、本地图/视频批、内存 PIL/ndarray、BCHW 张量。统一协议：`for paths, imgs, info in loader` 返回路径列表、图像列表（BGR `np.ndarray` 或 `LoadTensor` 的 tensor）、描述字符串。`build.load_inference_source` 根据 `check_source` 标志选用此处类；`SourceTypes` dataclass 标记源类型。顶层 **`LOADERS`** 元组供 `check_source` 识别已是 loader 的实例。

**图像尺寸与通道**：`channels=1` 时 `cv2_flag=IMREAD_GRAYSCALE`，读入后 `[..., None]` 保持 H×W×1；默认 3 为 BGR。**Tile**：`tile_batches` 按窗口裁切列表批，不复制整图。

## 逐段说明

### 导入（第 3–27 行）

`csv`/`glob` 解析清单；`Thread` 拉流；`tile_windows` 与训练 tile 同几何；`PIL_FALLBACK_SUFFIXES` 控制多线程解码可否并行。

---

### `SourceTypes`（第 30–54 行）

四个 bool 字段：`stream`、`screenshot`、`from_img`、`tensor`，默认全 False。

---

### `tile_batches`（第 57–62 行）

对 `images` 枚举出 `(source_index, (x1,y1,x2,y2))` 窗口列表；按 `batch` 步长切块；yield 裁切后的 numpy 列表、窗口元数据、是否最后一块。切片 `images[i][y1:y2,x1:x2]` 为视图/数组块。

---

### `LoadStreams`（第 65–253 行）

多路 RTSP/摄像头/YouTube；**daemon 线程** `update` 持续 `grab/retrieve`；`bs = 源个数`。

#### `__init__`（第 107–175 行）

- `cudnn.benchmark=True` 固定尺寸推理加速。
- `sources` 为文件则 UTF-8 读多行 URL；否则单元素列表。
- 每路：YouTube → `get_best_youtube_url`；纯数字 → 摄像头索引；Colab/Kaggle 禁 `0`。
- `VideoCapture` 打开；读宽高、fps、帧数（0→inf 流）；**首帧必读**成功；灰度则 BGR→GRAY 增维；起线程 `update`。
- 异常时 `close()` 再抛出。

#### `update`（第 177–203 行）

`n < frames-1` 循环；缓冲 `<30` 帧时 `grab`，每 `vid_stride` 次 `retrieve`；失败则黑图+重连 `cap.open(stream)`；`buffer=True` append，否则替换为单帧列表；满缓冲 `sleep(0.01)`。

#### `close`（第 205–217 行）

`running=False`，join 线程 5s，`cap.release()`。

#### `__iter__` / `__next__`（第 219–249 行）

`count` 从 -1 递增；每路等待非空缓冲，线程死则 `StopIteration`；`buffer` 模式 `pop(0)`，否则 `pop(-1)` 并 `clear()` 只留最新帧。

#### `__len__`（第 251–253 行）

返回 `bs`（源数量，非帧数）。

---

### `LoadScreenshots`（第 256–332 行）

依赖 **mss** 截屏；`source` 解析 `screen [id] [left top width height]`。

#### `__init__`（第 286–319 行）

`check_requirements("mss")`；参数分段设 `screen/left/top/width/height`；`monitor` 字典给 `grab`；`bs=1`，`fps=30`。

#### `__next__`（第 325–332 行）

`np.asarray(grab)[:,:,:3]` 去 alpha；可选灰度；返回单元素路径/图/info，含 LTWH 描述。

---

### `LoadImagesAndVideos`（第 335–526 行）

文件/目录/glob/txt/csv 清单；图+视频混排；`__len__` = `ceil(nf/bs)` **批次数**。

#### `__init__`（第 378–436 行）

- **txt/csv**：UTF-8；csv 可跳过 `source` 表头；相对路径相对清单父目录解析。
- 逐项 absolute path；glob/目录/单文件收集；按后缀分 `images`/`videos`；`files = images+videos`；`video_flag` 布尔列表。
- 有视频则 `_new_video` 第一个；`nf==0` 抛错。

#### `close`（第 438–441 行）

释放 `VideoCapture`。

#### `__next__`（第 448–504 行）

凑满 `bs` 张：

- **视频**：`grab`+`retrieve`，中间 `vid_stride-1` 次只 grab；失败则关 cap、计数+1，已有图则 **break 不跨视频拼 batch**。
- **图片**：若剩余≥8 且无非 PIL 回退后缀，**ThreadPool** 并行 `imread`；否则单张 `_append_image`。
- 列表耗尽：`StopIteration` 或返回尾部不足 batch。

#### `_append_image`（第 506–513 行）

`im0 is None` 警告跳过；否则追加 path/img/info 行号描述。

#### `_new_video`（第 515–522 行）

`frame=0`；`frames = ceil(frame_count/vid_stride)`；fps 保留 float。

#### `__len__`（第 524–526 行）

`ceil(nf/bs)`。

---

### `LoadPilAndNumpy`（第 529–630 行）

内存图单次 batch；`__next__` **只迭代一次**（`count==1` 即停）。

#### `__init__`（第 555–571 行）

包装为 list；空 list 抛错；`paths` 用 PIL `filename` 或 `image{i}.jpg`。

#### `_single_check`（第 573–614 行）

**PIL**：EXIF 旋转 274；I;16 右移 8 位到 uint8；转 L/RGB numpy；RGB→BGR。

**ndarray**：uint16 缩 8 位；浮点转 uint8；`atleast_3d`；校验 H×W×C 全非零。

通道对齐：1↔3 用 `cvtColor` 或 `repeat`；4 通道 BGRA 处理。

#### `__next__` / `__iter__`（第 620–630 行）

返回 paths、im0 列表、空 info 占位。

---

### `LoadTensor`（第 633–704 行）

输入 **BCHW**；`__next__` 同样只一轮。

#### `__init__`（第 656–666 行）

`_single_check` 后 `bs = shape[0]`。

#### `_single_check`（第 668–688 行）

4D 必选；3D 警告并 `unsqueeze(0)`；H/W 须非零且 **整除 stride 32**；`max>1` 浮点警告并 `/255`。

#### `__next__`（第 695–700 行）

返回 paths、**`torch.Tensor` 批**（非 list numpy）、info。

---

### `autocast_list`（第 707–740 行）

元素为路径/URL/PIL/ndarray：HTTP 用 requests+`Image.open`；EXIF 转置；不支持类型 `TypeError`。

---

### `get_best_youtube_url`（第 743–789 行）

`method` 为 `pytube`（pytubefix 最高 mp4 仅视频流）、`pafy`、或 `yt-dlp` 逆序 formats 选无音频 mp4；`check_requirements` 懒安装。

---

### `LOADERS`（第 792–793 行）

元组 `(LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)` — `LoadTensor` 不在内，因专门由 tensor 标志分支构造。