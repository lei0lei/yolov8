# `ultralytics/models/sam/modules/tiny_encoder.py`

本文件实现 **TinyViT** 轻量视觉主干（Microsoft 架构，LeViT/Swin 启发），供小型 SAM 权重（如 mobile/tiny 变体）作 `image_encoder`。结构为卷积 PatchEmbed、首 stage 的 MBConv `ConvLayer`、后续 stage 的窗口注意力 `BasicLayer`、末端与 SAM ViT 类似的 **neck**（1×1+3×3 卷积至 256 维）。`TinyViT.forward`/`forward_features` 输出与 `ImageEncoderViT` 相容的 `(B,256,h,w)` 特征图；`set_imgsz` 动态调整各层 `input_resolution`。

## 逐段说明

### `Conv2d_BN` 类（第 27–74 行）

#### `__init__`

`Conv2d`（无 bias）+ `BatchNorm2d`，BN weight 初始化为 `bn_weight_init`。

---

### `PatchEmbed` 类（第 77–123 行）

#### `__init__`（第 99–119 行）

两层 3×3 stride-2 `Conv2d_BN`，总下采样 4×；记录 `patches_resolution = (img/4, img/4)`。

#### `forward`（第 121–123 行）

`(B,3,H,W) → (B, embed_dim, H/4, W/4)`。

---

### `MBConv` 类（第 126–191 行）

#### `__init__` / `forward`

1×1 扩展 → depthwise 3×3 → 1×1 投影（末 BN weight=0）；残差 + 激活。`drop_path` 固定为 Identity。形状 `(B,C,H,W)` 不变。

---

### `PatchMerging` 类（第 194–253 行）

#### `forward`（第 239–253 行）

3D 输入先还原为 `(B,C,H,W)`；1×1 升维 → depthwise stride 卷积（部分 `out_dim` 不 stride）→ 1×1；flatten 为 `(B, H'W', out_dim)` 序列供 Transformer stage。

---

### `ConvLayer` 类（第 256–338 行）

#### `__init__` / `forward`

堆叠 `depth` 个 `MBConv`；可选 `downsample`（`PatchMerging`）。支持 gradient checkpoint。

---

### `MLP` 类（第 341–397 行）

#### `forward`

`LayerNorm → fc1 → act → drop → fc2 → drop`；token 形状 `(B,N,C)` 不变。

---

### `Attention` 类（第 400–512 行）

#### `__init__`（第 430–478 行）

预计算窗口内所有像素对的 attention bias 索引；可学习 `attention_biases (num_heads, n_offsets)`。

#### `train`（第 480–488 行）

评估模式缓存 `ab` 偏置矩阵；训练时删除缓存。

#### `forward`（第 490–512 行）

`norm → qkv` 拆 q,k,v；`(q@k^T)*scale + bias`；softmax；乘 v；`proj`。输入输出 `(B,N,C)`。

---

### `TinyViTBlock` 类（第 515–653 行）

#### `__init__`（第 541–592 行）

窗口 `Attention` + 深度可分离 `local_conv` + `MLP`；`drop_path` 为 Identity。

#### `forward`（第 594–634 行）

`x (b, hw, c)`：若分辨率等于 `window_size` 则直接注意力，否则 pad 并划窗 → 注意力 → 还原。残差后转为 `(b,c,h,w)`，`local_conv`，再展平为序列；`+ mlp(x)` 残差。输出 `(b, hw, c)`。

#### `extra_repr`（第 636–653 行）

返回超参字符串。

---

### `BasicLayer` 类（第 656–756 行）

#### `__init__`（第 679–746 行）

`depth` 个 `TinyViTBlock` + 可选 `downsample`。

#### `forward`（第 748–752 行）

顺序 block；末层可选 `PatchMerging` 输出降采样序列或特征图（取决于 downsample 输入格式）。

#### `extra_repr`（第 754–756 行）

打印 dim、resolution、depth。

---

### `TinyViT` 类（第 759–986 行）

#### `__init__`（第 787–897 行）

`patch_embed`；`layers[0]` 为 `ConvLayer`，其余为 `BasicLayer`（参数含 `embed_dims`、`depths`、`window_sizes`）；分类头 `norm_head`+`head`（推理特征路径可不使用）；`neck` 输出 256 通道。

#### `set_layer_lr_decay`（第 900–933 行）

为各参数设置 `lr_scale` 与 `param_name`（训练用）；`apply(_check_lr_scale)`。

#### `_init_weights`（第 935–945 行）

Linear/LayerNorm 初始化。

#### `no_weight_decay_keywords`（第 947–950 行）

返回 `attention_biases`。

#### `forward_features`（第 952–965 行）

`patch_embed` → 各 `layer`；末序列 reshape 为 `(B, C, h/4, w/4)` 空间布局（相对 patch 网格再 /4）；`neck` → `(B, 256, H', W')`。

#### `forward`（第 967–969 行）

调用 `forward_features`。

#### `set_imgsz`（第 971–986 行）

更新 `patches_resolution` 及各 layer/block 的 `input_resolution`；`downsample.interpol_size` 由 SAM3 等在子类中间接使用。