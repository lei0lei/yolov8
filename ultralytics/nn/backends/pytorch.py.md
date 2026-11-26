# `ultralytics/nn/backends/pytorch.py`

本文件实现原生 PyTorch 与 TorchScript 两条推理路径，是 `AutoBackend` 在 `.pt` 与 `.torchscript` 权重下的默认载体。`PyTorchBackend` 通过 `load_checkpoint` 或直接使用 `nn.Module`，支持 Conv+BN 融合、FP16、`end2end` 头切换，并暴露 `augment`/`embed` 等仅 `BaseModel` 才有的前向参数。`TorchScriptBackend` 用 `torch.jit._serialization.load` 反序列化追踪图，并从 zip/extra 中读取嵌入元数据。两者输出与训练模型前向一致（张量或张量列表），predictor 按 `task` 解析检测、分割等结构。

## 逐段说明

### `PyTorchBackend` 类（第 17–97 行）

#### `__init__`（第 24–46 行）

在调用 `BaseBackend.__init__` 前保存 `fuse`、`verbose`、`end2end_override`，后者可在融合前强制检测头 one2one/one2many 模式。

#### `load_model`（第 48–79 行）

- 路径则 `load_checkpoint(weight, device)` 取 `[0]`；已是 Module 则直接用。
- `end2end_override` 非空则写 `model.end2end`。
- `fuse` 且存在 `model.fuse`：Jetson JP5 先 `.to(device)`，再对 `BaseModel` 调用 `fuse(verbose=...)`。
- 同步 `kpt_shape`、`stride`（至少 32）、`names`（含 DDP `module`）、`colors`、`channels`（来自 yaml）。
- `half()`/`float()`；`requires_grad=False`；设置 `self.end2end`、`base_model=isinstance(unwrap, BaseModel)`。

#### `forward`（第 81–97 行）

非 `base_model` 时 `self.model(im, **kwargs)`。否则 `self.model(im, augment=..., embed=..., **kwargs)`。返回与模型 `forward` 相同（训练头原始 dict 或推理解码张量）。

### `TorchScriptBackend` 类（第 100–135 行）

#### `load_model`（第 107–124 行）

依赖 `torchvision` 反序列化；Torch 1.10–2.0 间禁用 texpr fuser 避免 trace 常量 bug。`jit.load` 到 `device`，`half`/`float`，`apply_metadata(read_metadata(weight))`。

#### `forward`（第 126–135 行）

`return self.model(im)`，输出为 trace 时固化的单张量或多输出列表。

## 与统一接口的对齐

| 阶段 | 行为 |
|------|------|
| 加载 | 元数据来自 checkpoint 内嵌 dict，不全时仍可从 `.pt` 属性读 `names`/`stride`。 |
| 前向 | 输入 BCHW [0,1]；输出与 `nn.tasks.BaseModel` 推理分支一致，分割为 `(det, proto)` 等。 |
| 对齐 | `base_model=True` 时 predictor 可传 `augment`/`embed`；TorchScript 无此扩展，输出形状由导出图决定。 |