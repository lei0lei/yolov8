# `ultralytics/models/sam/sam3/geometry_encoders.py`

几何提示工具：`Prompt` 维护序列优先的归一化 cxcywh 框与 mask/标签；`concat_padded_sequences` 合并右填充序列；`SequenceGeometryEncoder` 用线性/RoI/正弦位置三种可叠加编码，再可选 Transformer 与 CLS。

## 逐段说明

### `is_right_padded` / `concat_padded_sequences`（行 12–69）

校验 mask 排序不变性；按真实长度散射 seq2 到 seq1 之后位置；生成新 padding mask；可选返回 seq2 索引。

### `Prompt`（行 72）

**`__init__`**：全 None 则空 prompt；否则默认 labels 全 1、mask 全 False；形状与 device 断言。

**`append_boxes`**（行 130）：首框直接赋；否则 `concat_padded_sequences` 扩展 labels 与 embeddings。

### `SequenceGeometryEncoder`（行 174）

**`__init__`**：正负 `label_embed`；可选 `cls_embed`；三路 box 投影器至少选一；`final_proj`+norm；`encode` 为 layer 克隆列表；RoI 路径时 `img_pre_norm` 为 LayerNorm。

**`_encode_boxes`**（行 238）：
- direct：Linear(4→d_model)；
- pool：`xywh2xyxy` 乘宽高尺度，`roi_align`，Conv2d 压到向量相加；
- pos_enc：`pos_enc.encode_boxes` 再 Linear；
- 加 `label_embed(boxes_labels)`。

**`forward(geo_prompt, img_feats, img_sizes, img_pos_embeds)`**（行 291）：
- 取最粗 level seq-first 特征与 pos；
- pool 时 reshape 为 NCHW；
- `_encode_boxes` 后可选 concat CLS；
- `final_proj`+encode 层 cross 到图像 memory；
- 返回 `(final_embeds, final_mask)`。