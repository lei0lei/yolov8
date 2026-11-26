# `ultralytics/models/sam/modules/decoders.py`

本文件提供 SAM 与 SAM2 的**掩码解码头**：在 `TwoWayTransformer` / `SAM2TwoWayTransformer` 提供的双向注意力之后，把图像 embedding 上采样并与 mask token 生成的 hypernetwork 权重做点积，得到低分辨率掩码 logits，并预测 IoU 质量分。`MaskDecoder` 服务于原版 SAM（`SAMModel`）；`SAM2MaskDecoder` 增加高分辨率 FPN 特征融合、物体存在分数、动态稳定性多掩码回退，以及用于视频跟踪的 `sam_tokens_out` 对象指针。二者均与 `PromptEncoder` 的稀疏/稠密提示及 `transformer.py` 中的双向 Transformer 衔接。

## 逐段说明

### `MaskDecoder` 类（第 11–163 行）

#### `__init__`（第 40–80 行）

登记 `iou_token`、`mask_tokens`（数量为 `num_multimask_outputs+1`）、转置卷积 `output_upscaling`（`C→C/4→C/8` 空间 4×）、每 mask token 一条 `MLP` hypernetwork，以及 `iou_prediction_head`。通道维均为 `transformer_dim`（通常 256）。

#### `forward`（第 82–124 行）

调用 `predict_masks` 得全量 `num_mask_tokens` 个掩码。`multimask_output=True` 时取索引 `1:`（三个歧义掩码）；否则取 `0:1`（单掩码 token）。同步裁剪 `iou_pred`。返回 `masks (B,M,H',W')`、`iou_pred (B,M)`，其中 `H'=4H`、`W'=4W`（相对 embedding 网格）。

#### `predict_masks`（第 126–163 行）

**Token 拼接**：`output_tokens = [iou_token; mask_tokens]` 扩成 `(B, 1+num_mask_tokens, C)`，再与 `sparse_prompt_embeddings (B, N, C)` 在序列维拼接 → `tokens (B, T, C)`。

**图像侧**：`image_embeddings` 按 batch 维 `repeat_interleave` 与 token batch 对齐（每图多 prompt 时）；`src = repeat(image_emb) + dense_prompt_embeddings`，形状 `(B, C, H, W)`。`pos_src` 同样 repeat `image_pe`。

**Transformer**：`hs, src = transformer(src, pos_src, tokens)`。`hs[:,0]` 为 IoU token；`hs[:,1:1+num_mask_tokens]` 为 mask tokens。

**上采样与掩码**：`src` 从 `(B,C,HW)` 还原为 `(B,C,H,W)`，经 `output_upscaling` → `(B, C/8, 4H, 4W)`。每个 mask token 经对应 MLP 得 `hyper_in (B, num_mask, C/8)`，与展平的 `upscaled_embedding` 做 batch 矩阵乘 → `masks (B, num_mask, 4H, 4W)`。`iou_prediction_head(iou_token_out)` → `(B, num_mask)`。

---

### `SAM2MaskDecoder` 类（第 166–496 行）

#### `__init__`（第 210–295 行）

在 SAM1 解码器基础上：`pred_obj_scores` 时增加 `obj_score_token` 且输出 token 顺序为 `[obj, iou, masks...]`。可选 `conv_s0`/`conv_s1` 融合高分辨率 backbone 特征。IoU 头可 `sigmoid`。登记动态稳定性多掩码相关超参。

#### `forward`（第 297–364 行）

`predict_masks` 返回四元组。多掩码模式：`masks[:,1:]`、`iou[:,1:]`。单掩码且 `dynamic_multimask_via_stability` 且非训练：调用 `_dynamic_multimask_via_stability`。否则取 channel 0。根据 `multimask_output` 与 `use_multimask_token_for_obj_ptr` 选择 `sam_tokens_out` 为 `(B,3,C)` 或 `(B,1,C)`。返回 masks、iou、`sam_tokens_out`、`object_score_logits (B,1)`。

#### `predict_masks`（第 366–403 行）

与 SAM1 类似拼接 tokens；`repeat_image=False` 时要求 `image_embeddings.shape[0]==tokens.shape[0]`，不再 repeat。`image_pe` batch 维必须为 1（来自 `get_dense_pe()`），再 repeat 到 batch。

#### `predict_masks`（第 404–435 行，上采样与输出）

Transformer 后索引随 `pred_obj_scores` 偏移 `s`。高分辨率路径：第一次转置卷积后与 `feat_s1` 相加，第二次后与 `feat_s0` 相加；否则仅 `output_upscaling`。hypernetwork 点积生成 `masks`。`pred_obj_scores` 时用 `hs[:,0]` 预测物体分数，否则用常数 10 的 logits（sigmoid≈1）。返回含 `mask_tokens_out` 的四元组。

#### `_get_stability_scores`（第 437–442 行）

对 `mask_logits` 最后一维展平，计算 `>δ` 与 `>-δ` 像素面积比作为稳定性。

#### `_dynamic_multimask_via_stability`（第 444–496 行）

比较 token-0 单掩码稳定性；低于阈值则从 token 1–3 中选 IoU 最高的掩码。`torch.where` 在单掩码与最佳多掩码间按 batch 选择。输出 `(B,1,H,W)`、`(B,1)`。