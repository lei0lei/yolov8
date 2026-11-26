# `ultralytics/models/sam/modules/encoders.py`

本文件实现 SAM 系列的**编码端**：图像主干（ViT 与 SAM2 的 Hiera+FPN）、提示编码、以及视频记忆的 `MemoryEncoder`。`ImageEncoderViT` 与 `PromptEncoder` 组装为 `SAMModel`；`ImageEncoder`（trunk+`FpnNeck`）+ `Hiera` 作为 SAM2/SAM3 视觉主干；`MemoryEncoder` 把当前帧特征与预测掩码融合写入记忆库。大量底层块（`Block`、`PatchEmbed` 等）定义在 `blocks.py` 并在此导入复用。

## 逐段说明

### `ImageEncoderViT` 类（第 23–143 行）

#### `__init__`（第 48–131 行）

`PatchEmbed` 将图像变为 `(B, H', W', embed_dim)`；可选绝对位置 `pos_embed`；`depth` 个 `Block`（部分层全局注意力由 `global_attn_indexes` 控制）；`neck` 为 `1×1 Conv + LN + 3×3 Conv + LN`，输出通道 `out_chans`（默认 256）。

#### `forward`（第 133–143 行）

`x = patch_embed(x)` → `(B, gh, gw, embed_dim)`。若 `pos_embed` 与网格不一致（如 `set_imgsz`），双线性插值对齐后相加。逐 `Block` 处理。`neck(x.permute(0,3,1,2))` → `(B, out_chans, gh, gw)`。

---

### `PromptEncoder` 类（第 146–332 行）

#### `__init__`（第 175–213 行）

`PositionEmbeddingRandom` 用于点/框坐标；4 类 `point_embeddings` + `not_a_point_embed`；mask 下采样 CNN 将输入掩码映射到与图像 embedding 同空间尺寸；`no_mask_embed` 表示无 mask 输入。

#### `get_dense_pe`（第 215–231 行）

对 `image_embedding_size` 生成随机 PE 网格，`unsqueeze(0)` → `(1, embed_dim, H, W)`。

#### `_embed_points`（第 233–248 行）

坐标 +0.5 像素中心；可 pad 一点；`forward_with_coords` 得 `(B, N, C)`，按 label -1/0/1/2/3 加对应 embedding。

#### `_embed_boxes`（第 250–257 行）

框 reshape 为两角点，PE + 角点类型 embedding 2/3。

#### `_embed_masks`（第 259–265 行）

必要时双线性插值到 `mask_input_size`，经 `mask_downscaling` → `(B, embed_dim, H_emb, W_emb)`。

#### `_get_batch_size`（第 267–281 行）

从 points/boxes/masks 任一项推断 batch 大小，全无则 1。

#### `forward`（第 283–332 行）

拼接点、框到 `sparse_embeddings (B, N_sparse, C)`。有 mask 则 `dense` 为下采样结果，否则 `no_mask_embed` 广播到 `(B, C, H_emb, W_emb)`。返回稀疏 + 稠密对。

---

### `MemoryEncoder` 类（第 335–421 行）

#### `__init__`（第 361–387 行）

`MaskDownSampler` 把掩码下采样到与 `pix_feat` 对齐；`pix_feat_proj` 1×1；`Fuser`（两层 `CXBlock`）融合特征与掩码；`PositionEmbeddingSine`；`out_proj` 可选改通道。

#### `forward`（第 389–421 行）

除 `skip_mask_sigmoid` 外对 masks `sigmoid`；下采样后与 `pix_feat` 相加，经 `fuser`、`out_proj`。`position_encoding(x)` → 与特征同空间尺寸的 PE。返回 dict：`vision_features (B,out_dim,H,W)`、`vision_pos_enc` 为单元素 list。

---

### `ImageEncoder` 类（第 424–484 行）

#### `__init__`（第 448–470 行）

绑定 `trunk`（如 Hiera）与 `neck`（FpnNeck），校验 `channel_list` 一致；`scalp` 丢弃最低分辨率层数。

#### `forward`（第 472–484 行）

`features, pos = neck(trunk(sample))`；若 `scalp>0` 截断列表尾部。返回 dict：最高层为 `vision_features`，全部 FPN 级为 `backbone_fpn`，对应 `vision_pos_enc` 列表。

---

### `FpnNeck` 类（第 487–620 行）

#### `__init__`（第 512–568 行）

每 backbone 级 1×1 卷积投到 `d_model`；`PositionEmbeddingSine(256)`；`fpn_top_down_levels` 控制哪些层接收自顶向下融合；`fuse_type` 为 sum 或 avg。

#### `forward`（第 570–620 行）

自低分辨率向高分辨率遍历：侧向卷积 + 上采样 `prev_features` 相加（可选平均）。每级输出 `d_model` 特征图及正弦 PE。输入 `xs[i]` 为高分辨率→低分辨率顺序。

---

### `Hiera` 类（第 623–807 行）

#### `__init__`（第 656–764 行）

7×7 stride-4 `PatchEmbed`；窗口/全局注意力交错的 `MultiScaleBlock` 序列；stage 末输出多尺度特征列表 `channel_list`。

#### `_get_pos_embed`（第 766–773 行）

双三次插值背景 PE + tile 窗口 PE，返回 `(1, h, w, C)`。

#### `forward`（第 775–807 行）

`patch_embed` → `(B,H,W,C)`，加位置；逐 block；在 stage 末将 `permute` 为 `(B,C,H,W)` 加入 `outputs`。返回多尺度 list（高分辨率在前或按 `return_interm_layers`）。