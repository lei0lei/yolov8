# `ultralytics/models/sam/sam3/encoder.py`

SAM3 编码侧重 GroundingDINO 风格层：每层 self-attn 后 cross-attn 到 prompt（文本/几何）memory，支持 pre/post-norm 与 DAC 半 query 自注意力。`TransformerEncoder` 展平多尺度 FPN；`TransformerEncoderFusion` 在编码前将池化文本加到图像特征上。

## 逐段说明

### `TransformerEncoderLayer.__init__`（行 26）

默认 8 头 MHA self/cross；FFN 两线性+ReLU；三 LayerNorm、三 Dropout；保存位置编码注入位置与 `pre_norm` 旗标。

### `forward_post(...)`（行 81）

- self-attn：q=k=tgt(+query_pos 可选)；残差+norm1；
- cross：query/key 可加 pos；残差+norm2；
- FFN：linear-激活-dropout-linear；残差+norm3。

### `forward_pre(...)`（行 138）

- **dac**：拆半 tgt，仅前半 self-attn，再 cat 回；
- norm 在前；cross 与 FFN 同 post 语义但顺序为 pre-norm。

### `forward`（行 195）

按 `pre_norm` 选择 `forward_pre` 或 `forward_post`。

### `TransformerEncoder.__init__`（行 256）

克隆 layer；多 level 时 `level_embed`；可冻结；为每层设 `layer_idx`；`use_act_checkpoint`。

### `_prepare_multilevel_features`（行 287）

断言 level 数；逐 level flatten HW、拼 level_embed、cat 全图序列；有 mask 时 `get_valid_ratio` 堆叠，否则全 1 比例。

### `TransformerEncoder.forward`（行 327）

prepare 后逐层传入 `memory=prompt`（batch-first 转置由 fusion 侧处理）、`query_pos` 为 lvl pos、`tgt_key_padding_mask`；训练断言启用 act ckpt；返回 seq-first 的 output、mask、pos、valid_ratios。

### `TransformerEncoderFusion`（行 396）

可选 `text_pooling_proj`；`torch.compile` forward；`add_pooled_text_to_img_feat` 时用 `pool_text_feat` 广播加到各 level src；`super().forward` 时 prompt 转 `(B,seq,C)`；返回 dict 含 memory_text 等。

### `pool_text_feat`（行 511）

无 mask 则 dim0 mean；否则按有效 token 加权平均。