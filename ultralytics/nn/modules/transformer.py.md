# `ultralytics/nn/modules/transformer.py`

本文件提供 YOLO 与 RT-DETR 共用的 Transformer 组件：标准编码层、2D 位置编码（AIFI）、ViT 风格块、可变形注意力与完整可变形解码器栈。模块被 `head.RTDETRDecoder` 与部分骨干/neck 引用，不负责最终检测损失，仅做特征融合与 query 更新。导出友好设计体现在 MSDeformAttn 的张量秩限制与 AIFI 位置编码的 fp32 计算。公开符号由 `__all__` 列出，供 `parse_model` 与外部导入。

## 逐段说明

### 模块导入与 `__all__`（第 1–29 行）

导入 `conv.Conv`、`utils` 中 clone/反 sigmoid/可变形注意力 PyTorch 实现；导出九个公共类名。

### `TransformerEncoderLayer`（第 32–168 行）

标准 Encoder 层：`MultiheadAttention(batch_first=True)` + FFN + LayerNorm；支持 pre-norm / post-norm。

#### `__init__`（第 51–89 行）

要求 `torch>=1.9`；构造 `ma`、`fc1/fc2`、`norm1/norm2`、dropout、`GELU`、`normalize_before`。

#### `with_pos_embed`（第 91–94 行，静态）

`pos` 非空则 `tensor + pos`。

#### `forward_post`（第 96–120 行）

Post-LN：Q=K=src+pos，注意力残差+norm1，FFN 残差，返回 norm2(src)。

#### `forward_pre`（第 122–146 行）

Pre-LN：先 norm1 再注意力，再 norm2+FFN。

#### `forward`（第 148–168 行）

按 `normalize_before` 分发到 `forward_pre` 或 `forward_post`。

### `AIFI`（第 171–246 行）

继承 `TransformerEncoderLayer`，面向 `(B,C,H,W)` 特征图。

#### `__init__`（第 178–197 行）

调用父类构造，无额外参数。

#### `forward`（第 199–212 行）

展平为 `(B,H*W,C)`，用 2D sin-cos `pos_embed` 调父类，再 reshape 回 `(B,C,H,W)`。

#### `build_2d_sincos_position_embedding`（第 214–246 行，静态）

`embed_dim` 需被 4 整除；meshgrid 行主序；h/w 各 sin/cos 半维；fp32 乘 omega 以适配 CoreML。

### `TransformerLayer`（第 249–277 行）

ViT 简化块（无 LayerNorm）：`q/k/v` 线性 + `MultiheadAttention` + 两层 FC，均残差。

#### `forward`（第 267–277 行）

输入 `(seq, batch, c)`，自注意力与 FFN 各加残差。

### `TransformerBlock`（第 280–323 行）

完整 2D ViT 块：可选 `Conv(c1→c2)`、`linear` 可学习位置编码、`num_layers` 个 `TransformerLayer`。

#### `forward`（第 310–323 行）

`(B,C,H,W)` → `(H*W,B,C)` 加位置 → 串联 TransformerLayer → reshape 回 spatial。

### `MLPBlock`（第 326–351 行）

两层 Linear + 激活，经典 FFN 单片。

### `MLP`（第 354–420 行）

可配置层数、激活、可选输出 sigmoid、残差（需 `input_dim==output_dim`）、`out_norm`。

#### `forward`（第 405–420 行）

逐层 Linear+激活，末层无激活；可选残差与 `out_norm`；`sigmoid` 结尾。

### `LayerNorm2d`（第 423–460 行）

通道维 LayerNorm：permute 到 NHWC 调 `F.layer_norm` 再 permute 回来。

### `MSDeformAttn`（第 463–582 行）

多尺度可变形交叉注意力。

#### `__init__`（第 484–508 行）

`d_model % n_heads == 0`；线性层产生 offset/weight/value/out；`_reset_parameters`。

#### `_reset_parameters`（第 510–529 行）

offset bias 设为多头环状 grid 初值；其余 Xavier/constant 初始化。

#### `forward`（第 531–582 行）

`value_proj`、可选 mask 填零；展平 `n_levels*n_points` 以控制 trace 秩；`refer_bbox` 2 点或 4 点模式计算采样坐标；调用 `multi_scale_deformable_attn_pytorch`；`output_proj`。

### `DeformableTransformerDecoderLayer`（第 585–709 行）

自注意力 → `MSDeformAttn` 交叉注意力 → FFN，每层后 Norm/残差。

#### `forward_ffn`（第 656–667 行）

两层 Linear+ReLU+dropout，残差+norm3。

#### `forward`（第 669–709 行）

`embed` 经 self-attn；cross-attn 用 `refer_bbox.unsqueeze(2)` 与多尺度 `feats`；返回 FFN 输出。

### `DeformableTransformerDecoder`（第 712–798 行）

堆叠 `num_layers` 个 decoder layer；训练每层输出 bbox/cls；推理仅 `eval_idx` 层。

#### `__init__`（第 728–741 行）

`_get_clones(decoder_layer, num_layers)`；`eval_idx` 负数从末尾计。

#### `forward`（第 743–798 行）

`refer_bbox.sigmoid()` 迭代；每层 `pos_mlp(refer_bbox)` 为 query pos；`bbox_head[i]` 残差 refine；训练 stack 全部层 bbox/cls，eval 在 `eval_idx` break；返回 `(dec_bboxes, dec_cls)` 均为 stack 维 `(L,B,Q,*)`。