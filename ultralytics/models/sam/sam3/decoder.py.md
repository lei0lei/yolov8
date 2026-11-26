# `ultralytics/models/sam/sam3/decoder.py`

SAM3 解码器：每层 self-attn（可选 DAC、presence token）、文本 cross-attn、可变形图像 cross-attn（含 boxRPB 相对位置偏置）、FFN；迭代 `box_refine` 更新参考框。`TransformerDecoder` 堆叠多层，强制 `return_intermediate` 与 `box_refine`，输出各层隐状态与 ref box、可选 presence logits。

## 逐段说明

### `TransformerDecoderLayer.__init__`（行 22）

cross_attn 注入；可选 `ca_text`；self_attn、三层 norm、FFN；`with_pos_embed` 静态加 pos。

### `forward_ffn`（行 63）

标准 FFN 残差+norm3。

### `TransformerDecoderLayer.forward`（行 70）

- `_apply_self_attention` 更新 tgt/query_pos；
- **use_text_cross_attention**：对 memory_text MHA+残差+norm；
- **presence_token**：在 cross_attn_mask 前拼 1 列零 mask；
- **图像 cross_attn**：key/value 为 memory；残差+norm1；
- **FFN**：`forward_ffn` 到 memory dtype；
- 有 presence 时拆出 `presence_token_out`，余下为 object queries。

### `_apply_self_attention`（行 140）

- **dac**：o2o/o2m 拆分，仅 o2o 进 self-attn，再 cat；
- **presence**：cat 到 query 序列，query_pos 前插零；
- self-attn 后 dac 分支按 `dac_use_selfatt_ln` 决定 norm 时机；非 dac 直接 norm2。

### `TransformerDecoder.__init__`（行 183）

克隆 decoder layer 与可选 fine_layers；`query_embed`、MLP `bbox_embed`、可选 instance 分支与 reference_points；boxRPB 时 MLP 生成 x/y 偏置；presence token 嵌入与 head；`ref_point_head` 由正弦参考点 MLP；断言 roi_pooler None、return_intermediate、box_refine；可 compile；每层 `layer_idx`。

### `_get_rpb_matrix`（行 304）

xywh→xyxy；网格与框角差分；log/both 模式对数归一化；MLP 映射 x/y 再广播相加得 `(bs,n_heads,nq,HW)` bias。

### `TransformerDecoder.forward`（前半，行 349–429）

- 有 memory_mask 时断言无 boxRPB 冲突；
- **apply_dac**：默认 `self.dac`，tgt 与 reference_boxes repeat 2 倍；
- 初始化 reference：`reference_points` sigmoid 或沿用输入；
- **presence**：`presence_token` 扩 batch；
- **is_instance_prompt**：切换 instance bbox/norm 头。

### `TransformerDecoder.forward`（逐层循环，行 434–511）

- `reference_points_input` 乘 valid_ratios 展平多 level；
- `gen_sineembed_for_position` + `ref_point_head` → `query_pos`；
- boxRPB 时 `memory_mask = _get_rpb_matrix` 并 flatten 为 `(bs*n_heads,nq,HW)`；
- 调 layer，传入 text、DAC、presence、decoder_extra_kwargs；
- **box_refine**：`box_head(out_norm(output))` 或分 det/trk 头；+inverse_sigmoid(ref) 再 sigmoid；detach 更新 reference；非末层 append 中间 ref；
- 记录 `out_norm(output)`；presence 时 head+clamp 中间 logits；
- 返回 stack 的 intermediate、ref_boxes、presence_logits、presence_feats。

### 设计约束（init 末尾断言）

不支持 roi_pooler；必须中间层输出与 box refine。