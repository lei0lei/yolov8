# `ultralytics/models/sam/sam3/sam3_image.py`

`SAM3SemanticModel` 串联 VL 主干、几何编码器、Transformer 与可选分割头，完成文本（及框）grounding：编码→解码→点积或线性分类与框 refine→mask。`forward_grounding` 支持按 `max_text_batch` 分块以控显存；`set_classes` 缓存文本嵌入。

## 逐段说明

### `_update_out(out, out_name, out_value, auxiliary=True, update_aux=True)`（行 19）

主输出取末层；aux 时同步填充 `aux_outputs` 列表各层同名键。

### `SAM3SemanticModel.__init__(...)`（行 37）

保存各子模块与 DAC/o2m/点积打分/联合 presence 监督等旗标；校验 o2m 与 dac 关系；`text_embeddings`/`names` 缓存字典。

### `_encode_prompt(...)`（行 101）

可选 `prev_mask_pred` 加到最后层图像特征；`geometry_encoder` 得 geo；无视觉 prompt 则空张量 concat；返回 `prompt, prompt_mask`。

### `_run_encoder(...)`（行 132）

`transformer.encoder` 调 fusion 编码器，`encoder_out` 打包 memory、pos、mask、valid_ratios、prompt 前后状态。

### `_run_decoder(...)`（行 167）

`query_embed` 扩 batch；decoder 输入 memory 与文本 mask；输出转 batch-first；`_update_scores_and_boxes` 写 logits/框。

### `_update_scores_and_boxes(...)`（行 210）

点积或线性得类分数；`bbox_embed`+inverse_sigmoid 得 cxcywh 与 xyxy；可选 presence 与联合分数 clamp；`_update_out` 写 pred_logits/boxes（仅 o2o 查询数）。

### `_run_segmentation_heads(...)`（行 260）

有 head 时按 `o2m_mask_predict` 选 query 子集，调用 head 并区分 instance_keys 是否写 aux。

### `forward_grounding(...)`（行 288）

- 仅文本超 batch 且无几何时递归分块 `cat` 结果；
- `SAM2Model._prepare_backbone_features` 准备多 level 特征；
- 按 `text_ids` 索引 language 特征；几何 prompt 则 concat 文本与 geo；
- profiler 段：encoder、decoder、seg head；
- 返回含检测与 mask 的 `out` dict。

### `set_classes` / `set_imgsz`（行 366–373）

文本嵌入经 `backbone.forward_text`；图像尺寸设于 backbone。