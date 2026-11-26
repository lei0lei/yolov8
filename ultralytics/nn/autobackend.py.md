# `ultralytics/nn/autobackend.py`

本模块在预测与基准测试路径上统一多种导出格式的推理后端：通过 `AutoBackend` 根据权重路径或 URL 选择具体 `nn.backends.*` 实现，并完成 dtype、布局、批大小与类名元数据的规范化。模块级函数 `check_class_names`、`default_class_names` 被 exporter、`files.py`、`YOLOEModel.set_vocab` 等复用。主要调用方包括 `engine/predictor.py`（加载非 `.pt` 或托管权重）、`tests/test_python.py`、以及 `engine/exporter.py` 在验证导出结果时的对照推理。`AutoBackend` 本身继承 `nn.Module`，但核心计算委托给 `self.backend`；多数属性经 `__getattr__` 透明转发（`model`、`device`、`task`、`names`、`stride`、`metadata` 等）。

## 导入说明（行 3–43）

- **标准库**：`platform`、`deepcopy`、`Path`、`Any` 类型。
- **NumPy / PyTorch**：数组与张量互转、设备选择。
- **Ultralytics 工具**：`LOGGER`、`check_suffix`、`is_url`、`Colors`、`select_device`、`smart_inference_mode`、`TORCH_*` 版本门闩。
- **`backends` 包**：各格式 `*Backend` 类，由 `_BACKEND_MAP` 键名映射。

## 逐段说明

### 函数 `check_class_names(names)`（行 46–76）

输入为 list 则 `dict(enumerate(names))`。对 dict：`int(k)` 键排序、值转 `str`；校验非空、`max(keys) < n`。若 `names[0]` 以 `n0` 开头（ImageNet synset），从 `ImageNet.yaml` 的 `map` 换为人类可读名。返回规范化的 `dict[int,str]`。

### 函数 `default_class_names(data, nc=999)`（行 79–97）

若提供 `data` 路径，尝试 `YAML.load(check_yaml(data))["names"]`；任意异常则忽略。失败则生成 `{i: f"class{i}" for i in range(nc)}`。

### 类 `AutoBackend`（行 100–467）

#### 类属性 `_BACKEND_MAP`（行 160–188）

将格式字符串（`pt`、`onnx`、`engine`…）映射到后端类；`dnn` 与 `onnx` 共享 `ONNXBackend`；`nxp` 映射 `LiteRTBackend` 等。

#### 类属性 `INTEL_FORMATS`（行 189–190）

`openvino`、`litert` 跳过 `select_device` 的通用 GPU 回退逻辑（Intel 设备映射 `xpu`/`npu`）。

#### `__init__(...)`（行 192–291）

**格式判定（行 219–221）**：传入 `nn.Module` 则格式为 `pt`，否则 `_model_type(model, dnn)`。

**设备（行 222–241）**：非 Intel 格式时 `select_device`；若已是 inference 模式 tensor 的 Module，则 `deepcopy` 以免融合破坏。`fp16` 仅在支持的后端且非 CPU PyTorch 时保持为真。若 CUDA 可用但格式不在白名单，强制 `cpu`（避免不支持的 GPU 路径）。

**后端构造（行 243–260）**：未知格式抛 `TypeError` 并列出 `export_formats()`；`pt` 额外传入 `fuse`、`verbose`、`end2end`；部分 TF/ONNX 变体传 `format` 关键字。实例化 `self._BACKEND_MAP[format](model, **backend_kwargs)`。

**channels_last（行 262–279）**：仅 `pt` 格式在 CPU+x86+MKLDNN 时默认开启（Linux/Windows）；否则警告并忽略。

**元数据（行 281–291）**：`nhwc` 标记 CoreML/TF 等布局；`self.format` 赋值；无 `names` 时用 `default_class_names(data)`；`check_class_names` 规范化；空类名警告；`Colors.parse` 缓存绘图色。

#### `__getattr__(name)`（行 293–310）

若 `backend` 已有属性则返回，否则走 `Module` 默认（可能 `AttributeError`）。

#### `forward(im, augment, embed, **kwargs)`（行 312–362）

**大批切分（行 330–339）**：非 CoreML/IMX 且 batch 大于导出 batch、且 NMS 或静态图时，按 `self.batch` `split` 递归 `forward` 再 `cat`（列表输出则 zip 拼接）。

**输入变换（行 340–345）**：`nhwc` 时 `permute(0,2,3,1)`；后端 `fp16` 且输入非 half 则 `half()`；静态 batch 不足则用零张量 pad 到 `self.batch`。

**调用（行 347–354）**：仅 `pt` 传 `augment`、`embed` 与 `kwargs` 给学生式 `backend.forward`。

**输出（行 353–362）**：去掉 pad 行；segment 且 `names==999` 时从输出形状反推 `nc`；单元素 list 转单个 tensor；`from_numpy` 统一到 `self.device`。

#### `from_numpy(x)`（行 364–375）

`ndarray` → `torch.as_tensor(..., device=self.device)`；已有 Tensor 则 `.to(device)`；其它类型原样返回。

#### `warmup(imgsz, im)`（行 377–400）

非 `end2end` 时惰性 import `torchvision`（NMS 路径）；在 GPU 或特定格式上分配/复用 dummy 输入，循环 1–2 次 `forward`；构造随机框并调用 `non_max_suppression` 预热 NMS 内核。

#### `_model_type(p, dnn)`（行 402–440）

静态：对照 `export_formats()["Suffix"]` 在文件名中找**最靠后且最长**匹配后缀；`.pt` 记为 `pt`；`onnx`+`dnn=True` → `dnn`；无匹配且 URL 为 http/grpc → `triton`。目录类后缀如 `*_openvino_model/` 由 exporter 表格定义。

#### `eval()`（行 442–446）

若 `backend.model` 有 `eval` 则调用；返回 `super().eval()` 链式结果。

#### `_apply(fn)`（行 448–467）

父类 `_apply` 后对 `backend.model`（若为 `Module`）同样 `_apply`，并据首个参数更新 `backend.device` 与 `backend.fp16` 标志。