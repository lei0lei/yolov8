# `ultralytics/models/sam/modules/blocks.py`

本文件是 SAM 视觉与解码子系统的**共享积木库**：DropPath、掩码下采样、ConvNeXt 融合块、双向 Transformer 的 SAM2 变体、RoPE 注意力、Hiera/ViT 用的多尺度与窗口注意力、位置编码与 PatchEmbed 等。`encoders.py` 的 ViT/Hiera/记忆编码、`decoders.py` 间接通过 `SAM2TwoWayTransformer`、`memory_attention.py` 的 RoPE 均依赖此处实现。设计上将可复用 NN 模块与 `utils.py` 中的张量算子配合，避免在 `sam.py` 中重复定义 backbone 细节。

## 逐段说明

### `DropPath` 类（第 20–51 行）

#### `__init__` / `forward`

训练时按 `drop_prob` 对 batch 维采样 Bernoulli mask，可选除以 `keep_prob` 保持期望；推理或 `drop_prob=0` 恒等。输入输出形状相同。

---

### `MaskDownSampler` 类（第 54–125 行）

#### `__init__`（第 75–113 行）

按 `total_stride` 与 `stride` 堆叠「通道扩张式」stride 卷积 + `LayerNorm2d` + 激活，末层 1×1 投到 `embed_dim`；可选 `interpol_size` 先双线性对齐掩码尺寸。

#### `forward`（第 115–125 行）

可选插值后过 `encoder`。典型：`(B,1,H,W) → (B, embed_dim, H/16, W/16)`（由层数决定）。

---

### `CXBlock` 类（第 128–210 行）

#### `__init__`（第 155–194 行）

Depthwise conv + LN + 两个 Linear（4× 扩展 GELU MLP）+ 可选 layer scale + DropPath。

#### `forward`（第 196–210 行）

`(N,C,H,W)`：dwconv → norm → `(N,H,W,C)` 上 MLP → 回 `(N,C,H,W)` + 残差。形状不变。

---

### `Fuser` 类（第 213–258 行）

#### `__init__` / `forward`

`num_layers` 个深拷贝的 `layer`；可选 1×1 `proj`。顺序应用，形状不变。

---

### `SAM2TwoWayAttentionBlock` 类（第 261–313 行）

#### `__init__`（第 289–313 行）

继承 `TwoWayAttentionBlock`，将 `MLPBlock` 换为 2 层 `MLP`（与 SAM2 解码器一致）。

---

### `SAM2TwoWayTransformer` 类（第 316–379 行）

#### `__init__`（第 345–379 行）

调用父类构造后，用 `SAM2TwoWayAttentionBlock` 列表替换 `self.layers`。`forward` 继承父类：图像 `(B,C,H,W)` ↔ `(B,HW,C)` 与 prompt tokens 双向更新。

---

### `RoPEAttention` 类（第 382–455 行）

#### `__init__`（第 406–420 行）

预计算 `freqs_cis`（轴向 RoPE）；`rope_k_repeat` 用于 memory 比 query 长的交叉注意力。

#### `forward`（第 422–455 行）

Q/K/V 投影并分头；按序列长重算 `freqs_cis`；对前 `num_k_rope` 个 key 应用 `apply_rotary_enc`；`F.scaled_dot_product_attention`；合并头与 `out_proj`。输出 `(B, N_q, embedding_dim)`。

---

### `do_pool` 函数（第 458–470 行）

`(B,H,W,C) → (B,C,H,W)`，pool（如 MaxPool），再 permute 回；可选 norm。用于注意力内 query 下采样。

---

### `MultiScaleAttention` 类（第 473–549 行）

#### `__init__`（第 502–521 行）

单线性 `qkv`；可选 `q_pool` 在注意力前降低 query 网格分辨率。

#### `forward`（第 523–549 行）

`x (B,H,W,C)`：reshape 为 token；q 可经 pool 使 `H,W` 变小；SDPA 在 head 维；`proj`。输出 `(B, H', W', dim_out)`。

---

### `MultiScaleBlock` 类（第 552–663 行）

#### `__init__`（第 582–628 行）

Norm → 可选 window partition → `MultiScaleAttention` → unpartition → 残差 → MLP 残差。stage 切换时 `q_stride` 触发 query pool，维数不匹配时 `proj`+pool 处理 shortcut。

#### `forward`（第 630–663 行）

见上；窗口分支用 `window_partition`/`window_unpartition`。输出 `(B, H', W', dim_out)`（下采样时空间尺寸可减半）。

---

### `PositionEmbeddingSine` 类（第 666–777 行）

#### `__init__`（第 693–712 行）

正弦编码半维 `num_pos_feats`；`cache` 按 `(H,W)` 缓存网格 PE。

#### `_encode_xy`（第 714–727 行）

1D 正弦/余弦编码 x、y 坐标向量。

#### `encode_boxes` / `encode_points`（第 729–744 行）

框/点批量编码，点格式含 label 维。

#### `forward`（第 746–777 行）

对特征图 `(B,C,H,W)` 生成 `(B, 2*num_pos_feats, H, W)` 位置图（y 与 x 编码 concat 于通道维）。

---

### `PositionEmbeddingRandom` 类（第 780–832 行）

#### `__init__` / `_pe_encoding`

可学习高斯矩阵将 `[0,1]^2` 坐标映到 sin/cos 特征。

#### `forward`（第 818–825 行）

网格 `size (h,w)` → `(C, h, w)`。

#### `forward_with_coords`（第 827–832 行）

按 `image_size` 归一化绝对坐标后编码 → `(B, N, C)`。

---

### `Block` 类（第 835–920 行）

#### `__init__`（第 861–903 行）

SAM ViT 块：`REAttention` + `MLPBlock`；`window_size>0` 时窗口注意力。

#### `forward`（第 905–920 行）

norm → 可选 window → `attn` → unwindow → 残差 → MLP 残差。`(B,H,W,C)` 不变。

---

### `REAttention` 类（第 923–1000 行）

#### `__init__`（第 949–983 行）

可选 `rel_pos_h/w` 参数表。

#### `forward`（第 985–1000 行）

QKV 线性；缩放点积注意力；`use_rel_pos` 时 `add_decomposed_rel_pos`；softmax @ V；`proj`。形状 `(B,H,W,C)` 不变。

---

### `PatchEmbed` 类（第 1003–1052 行）

#### `__init__` / `forward`

`Conv2d` 切片投影；`forward`：` (B,C,H,W) → (B, H', W', embed_dim)`。