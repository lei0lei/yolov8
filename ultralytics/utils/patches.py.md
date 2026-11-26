# `ultralytics/utils/patches.py`

本文件集中对 OpenCV、PIL、PyTorch 的猴子补丁与上下文管理器：跨平台读写含非 ASCII 路径图像、EXIF 方向、16 位与多页 TIFF、HEIF 懒加载、`torch.load`/`torch.save` 行为、ONNX 导出 arange 与 Dynamo 兼容，以及临时覆盖训练配置。模块加载时即用 `image_open` 替换 `Image.open`。目标是默认行为与 Ultralytics 数据管线一致（图像直立、标签与显示帧对齐）。

## 逐段说明

### 模块常量（第 18～20 行）

- `_imshow` 备份原 `cv2.imshow`；`PIL_FALLBACK_SUFFIXES` 列出需 PIL 解码的扩展名。

### `imread`（第 23～70 行）

- **第 46～52 行**：`np.fromfile` 读字节；`FileNotFoundError`/`OSError` 或空文件返回 `None`。
- **第 54～59 行**：非灰度且 tiff/tif：`imdecodemulti` UNCHANGED；尺寸一致且 UNCHANGED 可 stack 多页为通道；否则取第一页，非 UNCHANGED 丢 alpha。
- **第 60～61 行**：仍无图且为 AVIF/HEIC 等则 `_imread_pil`。
- **第 62～67 行**：`imdecode`；WEBP 或 UNCHANGED 视为 stored，可对 EXIF 调 `_exif_transpose`。
- **第 68～69 行**：uint16 且非 ANYDEPTH/UNCHANGED 规则时右移 8 位转 uint8。
- **第 70 行**：二维数组扩为 H×W×1 通道。

### `image_open`（第 78～107 行）

- **第 95～96 行**：已注册 HEIF 则直接调备份 `_image_open`。
- **第 97～98 行**：try 原生打开。
- **第 99～107 行**：仅 `UnidentifiedImageError` 时 `check_requirements("pi-heif")`、`register_heif_opener`、设标志、重试。

### `Image.open` 赋值（第 110 行）

- 全局替换为 `image_open`。

### `_imread_pil`（第 113～129 行）

- **第 124～127 行**：`ImageOps.exif_transpose(Image.open)`；灰度转 L 数组；否则 RGB→BGR。
- **第 128～129 行**：异常返回 `None`。

### `_exif_transpose`（第 132～153 行）

- **第 134～139 行**：PNG 手动找 eXIf chunk，无则不改。
- **第 140～141 行**：非 JPEG/WebP 且非上述 PNG 分支则直接返回（TIFF 由 cv2 处理）。
- **第 142～146 行**：PIL 读 EXIF tag 274，失败返回原图。
- **第 147～152 行**：按 orientation 值 swapaxes、水平/垂直翻转。
- **第 153 行**：`ascontiguousarray`。

### `imread_unicode`（第 156～176 行）

- 仅 `fromfile`+`imdecode`，无 TIFF/HEIF/维数扩展；供 Windows 上 patch `cv2.imread`。

### `imwrite`（第 179～201 行）

- `imencode(后缀, img, params)[1].tofile`；异常返回 `False`，成功 `True`。

### `imshow`（第 204～221 行）

- 窗口名 `encode("unicode_escape").decode()` 后调 `_imshow`。

### `torch_load`（第 228～252 行）

- **第 247～250 行**：`TORCH_1_13` 时默认 `weights_only=False`；更老版本 pop 该 kwarg。
- **第 252 行**：调用原生 `torch.load`。

### `torch_save`（第 255～275 行）

- **第 269～275 行**：最多 4 次尝试 `_torch_save`；`RuntimeError` 则除最后一次外 `sleep(2**i/2)`。

### `arange_patch`（第 278～302 行）

- **第 289～300 行**：仅 `dynamic and quantize==16 and fmt=="onnx"` 时包装 `torch.arange` 为创建后再 `.to(dtype)`；finally 还原。
- **第 301～302 行**：否则空 `yield`。

### `onnx_export_patch`（第 305～323 行）

- **第 310～321 行**：`TORCH_2_9` 时 patch `torch.onnx.export` 强制 `dynamo=False`；finally 还原。

### `override_configs`（第 326～346 行）

- **第 337～344 行**：有 overrides 则 `copy(args)`、setattr 每项、`yield args`、`finally` 还原 `__dict__`。
- **第 345～346 行**：无 overrides 直接 `yield args`。