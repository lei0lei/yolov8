# `ultralytics/models/sam/sam3/maskformer_segmentation.py`

Mask2Former 风格分割头：`PixelDecoder` 自顶向下 FPN 融合；`MaskPredictor` 用对象 query 与像素嵌入 einsum 得 mask；`SegmentationHead`/`UniversalSegmentationHead` 接 backbone 与 encoder 状态，后者额外输出语义图与 presence logits，并可用 prompt cross-attn 精炼 encoder 特征。

## 逐段说明

### `LinearPresenceHead`（行 17）

`Sequential(Identity, Identity, Linear(d,1))` 兼容旧权重；`forward` 忽略 hs/prompt 直传 Linear。

### `MaskPredictor`（行 30）

`MLP` 得 mask 嵌入；按 query 维 3/4 与 pixel 3/4 维选 einsum 式 `bqhw` 或 `lbqhw`。

### `SegmentationHead`（行 57）

`use_encoder_inputs` 时用 encoder 最后一 level 视觉 token 替换 backbone 顶层；`no_dec` 则 Conv2d 1 通道；否则 `MaskPredictor`；`act_ckpt` 可 checkpoint pixel_decoder；`_embed_pixels` 在 bs=1 时 squeeze 索引 `[0]`。

**`forward`**：embed pixels 后按 aux/no_dec 选 query 层，返回 `pred_masks`。

### `PixelDecoder`（行 143）

自顶向下：`prev_fpn` 上采样加当前 FPN level，`shared_conv` 时共用 layer 0；Conv+GroupNorm+ReLU；可 compile。

### `UniversalSegmentationHead`（行 191）

强制 `use_encoder_inputs=True`；可选 `presence_head`（点积或线性）；`cross_attend_prompt` 为 MHA；`semantic_seg_head` 1 通道；`instance_seg_head` 1×1 到 d_model。

**`forward`**：
- 可选 prompt cross-attn 残差到 `encoder_hidden_states`；
- presence：encoder mean pool 经 presence_head；
- `instance_embeds = instance_seg_head(pixel_embed)` 再 mask 预测；
- 返回 `pred_masks`、`semantic_seg`、`presence_logit`。