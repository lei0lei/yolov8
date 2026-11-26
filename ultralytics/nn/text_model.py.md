# `ultralytics/nn/text_model.py`

本模块为视觉-语言任务（YOLO-World、YOLOE、FastSAM 等）提供统一的文本编码抽象与多种后端实现。对外接口包括抽象基类 `TextModel`、具体实现 `CLIP`、`MobileCLIP`、`MobileCLIPTS`，以及工厂函数 `build_text_model`。`nn/tasks.py` 中 `WorldModel`、`YOLOEModel` 在 `get_text_pe` / `set_classes` 时调用 `build_text_model`；`tests/test_python.py` 与 `models/fastsam/predict.py` 直接使用 `MobileCLIPTS`。模型权重默认下载到 `WEIGHTS_DIR/clip` 或经 `attempt_download_asset` 拉取 TorchScript 文件。编码输出均为 L2 归一化特征（`MobileCLIPTS` 的文本路径由 TS 内部完成归一化），便于与图像特征做点积相似度。

## 导入说明（行 3–19）

- **标准库**：`annotations` 启用前向类型注解；`abc.abstractmethod` 定义接口；`pathlib.Path` 检查本地权重文件。
- **PyTorch / PIL**：`torch`、`nn`、`Image` 为推理与张量设备放置。
- **Ultralytics**：`WEIGHTS_DIR`、`checks.check_requirements` 做惰性依赖安装；`smart_inference_mode` 包裹推理方法以关闭梯度。
- **可选 CLIP**：`try/import clip`，失败则安装 Ultralytics 维护的 `CLIP.git` 再导入。

## 逐段说明

### 类 `TextModel`（行 22–43）

#### `__init__`（行 33–35）

调用 `nn.Module` 基类初始化，无额外状态；子类负责加载实际编码器。

#### `tokenize`（行 37–39）

抽象方法：子类将字符串或字符串列表变为 token 张量，形状一般为 `(batch, context_length)`。

#### `encode_text`（行 41–43）

抽象方法：接受已 tokenize 的张量与可选 `dtype`，返回归一化文本特征向量。

### 类 `CLIP`（行 46–160）

OpenAI CLIP 文本（兼图像）编码封装。

#### `__init__(size, device)`（行 70–84）

调用 `clip.load(size, device, download_root=WEIGHTS_DIR/clip)`，得到 `self.model` 与 `self.image_preprocess`；`self.to(device)` 迁移参数；`eval()` 固定为推理模式。结果：实例可在给定设备上对文本/图像编码。

#### `tokenize(texts, truncate=True)`（行 86–104）

使用 `clip.tokenize`；`truncate=True` 时截断超长文本避免 77 token 上限报错；返回值 `.to(self.device)`。结果：`(N, 77)` 整型 token 张量。

#### `encode_text(texts, dtype=float32)`（行 106–129）

`@smart_inference_mode` 下调用 `self.model.encode_text`，转 `dtype`，再按 L2 在最后一维归一化。结果：`(batch, embed_dim)` 单位向量，典型 512 维。

#### `encode_image(image, dtype)`（行 131–160）

若输入为 PIL，`image_preprocess` 后 `unsqueeze(0)` 并送设备；否则视为已预处理张量。`encode_image` 后同样 L2 归一化。结果：`(batch, embed_dim)` 图像特征。

### 类 `MobileCLIP`（行 163–252）

Apple MobileCLIP 完整 PyTorch 实现（非默认工厂路径，但类仍可用）。

#### `__init__(size, device)`（行 188–215）

惰性 `import mobileclip`（失败则装 Ultralytics fork）。`config_size_map` 将 `blt` 映射到配置 `b`。权重文件 `mobileclip_{size}.pt` 不存在时从 Apple CDN 下载。`create_model_and_transforms` 建模型，`get_tokenizer` 建分词器，`eval()`。结果：可训练的 MobileCLIP 文本塔。

#### `tokenize(texts)`（行 217–230）

仅接受 `list[str]`，经 `self.tokenizer` 得到张量并放到 `self.device`。

#### `encode_text(texts, dtype)`（行 232–252）

前向文本塔，`to(dtype)` 后 L2 归一化（与 CLIP 相同策略）。

### 类 `MobileCLIPTS`（行 255–353）

TorchScript 版 MobileCLIP/CLIP 文本编码器，为 `build_text_model` 中 `mobileclip` / `mobileclip2` 的默认实现。

#### `__init__(device, weight="mobileclip_blt.ts")`（行 277–299）

`torch.jit._serialization.load(attempt_download_asset(weight))` 加载 `self.encoder`。若模块带 `encode_image`，则设置 `Resize(224)+ToTensor` 预处理，CUDA 上 `half()` 否则 `float()`。分词器复用 `clip.clip.tokenize`。结果：轻量 JIT 推理路径。

#### `tokenize(texts, truncate=True)`（行 301–319）

与 CLIP 相同 tokenize API，支持 `truncate=False` 严格长度检查。

#### `encode_text(texts, dtype)`（行 321–340）

直接 `self.encoder(texts)`，注释说明 TS 内已含归一化，仅 `to(dtype)`。

#### `encode_image(image, dtype)`（行 342–353）

要求 `(B,3,224,224)` 预处理张量；调用 `encode_image` 并匹配 encoder 参数 dtype。

### 函数 `build_text_model(variant, device)`（行 356–383）

解析 `variant` 为 `base:size`（`split(":")`）。

- **`base == "clip"`**：返回 `CLIP(size, device)`，`size` 如 `ViT-B/32`。
- **`base == "mobileclip"`**：忽略 size，返回 `MobileCLIPTS(device)`，默认 `mobileclip_blt.ts`。
- **`base == "mobileclip2"`**：`MobileCLIPTS(device, weight="mobileclip2_b.ts")`。
- **其它**：`ValueError` 列出支持的 base 名。

`YOLOEModel` 的 YAML `text_model` 字段（默认 `mobileclip:blt`）经此工厂实例化。