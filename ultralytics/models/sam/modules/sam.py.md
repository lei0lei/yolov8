# `ultralytics/models/sam/modules/sam.py`

本文件是 Ultralytics **SAM / SAM2 / SAM3 的顶层 `nn.Module` 组装**：`SAMModel` 仅持有 ViT 编码器、提示编码器与 `MaskDecoder`（前向在 `SAMPredictor`）；`SAM2Model` 在图像主干之外串联 `memory_attention`、`memory_encoder`，并实现视频 `track_step` 流水线（记忆条件特征 → SAM 头 → 写记忆）；`SAM3Model` 继承 SAM2 但替换解码头 Transformer 与 `forward_image` 入口。常量 `NO_OBJ_SCORE=-1024` 用于“无物体”时掩码 logits 抑制。Meta 版权声明保留于文件头。

## 逐段说明

### 模块常量 `NO_OBJ_SCORE`（第 21–22 行）

大负数，叠加到无物体情形的低分辨率掩码 logits 上。

---

### `SAMModel` 类（第 25–90 行）

#### `__init__`（第 55–81 行）

注册 `image_encoder`、`prompt_encoder`、`mask_decoder` 及 ImageNet 风格 `pixel_mean`/`pixel_std` buffer（`(3,1,1)`）。

#### `set_imgsz`（第 83–90 行）

若 backbone 支持则转发；更新 `prompt_encoder` 的 `input_image_size`、`image_embedding_size`（`//16`）、`mask_input_size`（4× embedding）；`image_encoder.img_size`。

---

### `SAM2Model` 类（第 93–1038 行）

#### `__init__`（第 171–351 行，骨干与记忆）

保存 `image_encoder`、`use_high_res_features_in_sam`、`num_feature_levels`（高分辨率为 3）。对象指针相关：`use_obj_ptrs_in_encoder` 时建 `mask_downsample`；时间编码与 `no_mem_embed`/`maskmem_tpos_enc` 等可学习参数；`memory_attention`、`memory_encoder` 及 `mem_dim`（可能经 memory encoder 压缩）。

#### `__init__`（第 351–351 行，SAM 头与编译）

`_build_sam_heads()`；可选 `torch.compile(image_encoder.forward)`。

#### `device` 属性（第 353–356 行）

`next(self.parameters()).device`。

#### `forward`（第 358–363 行）

恒抛 `NotImplementedError`，引导使用 `SAM2VideoPredictor`。

#### `_build_sam_heads`（第 365–410 行）

创建 `PromptEncoder`（mask_in_chans=16）与 `SAM2MaskDecoder`（`SAM2TwoWayTransformer` depth=2）。若 `use_obj_ptrs_in_encoder`：`obj_ptr_proj`（可选 MLP）；`obj_ptr_tpos_proj` 可选线性。

---

#### `_forward_sam_heads`（第 412–494 行，提示编码）

输入 `backbone_features (B,C,H,W)`。点提示缺省时 pad 空点 label -1。mask 提示可先双线性插值到 `mask_input_size`（antialias）。`sam_prompt_encoder` → `sparse (B,N,C)`、`dense (B,C,H,W)`。

#### `_forward_sam_heads`（第 495–528 行，解码与上采样）

`sam_mask_decoder` 得低分辨多掩码、IoU、`sam_output_tokens`、`object_score_logits`。`pred_obj_scores` 时无物体帧将低分辨掩码置 `NO_OBJ_SCORE`。双线性上采样到 `image_size` 得 `high_res_multimasks`。多掩码模式下按 argmax IoU 选最佳单掩码；`sam_output_token` 同步选取。

#### `_forward_sam_heads`（第 529–550 行，对象指针）

`obj_ptr = obj_ptr_proj(sam_output_token)`；按 `soft_no_obj_ptr`/`fixed_no_obj_ptr` 与 `no_obj_ptr` 混合。返回 7 元组（低/高分辨多掩码与最佳掩码、IoU、obj_ptr、object_score_logits）。

---

#### `_use_mask_as_output`（第 552–597 行）

掩码输入直接映射为 logits（`×20-10`）；下采样得低分辨；假 IoU=1。可选经 SAM 头从 `mask_downsample` 得 `obj_ptr`；由掩码是否全零推断 `object_score_logits` 与 `no_obj_ptr` 混合。

#### `forward_image`（第 599–607 行）

`image_encoder` → 若高分辨率 SAM 则预计算 `conv_s0/s1` 于 FPN 0/1 层。

#### `_prepare_backbone_features`（第 609–627 行）

可选 expand batch；取最后 `num_feature_levels` 级 FPN；`flatten+permute` 为 `(HW,B,C)` 列表与 `feat_sizes`。

---

#### `_prepare_memory_conditioned_features`（第 629–649 行）

`num_maskmem==0` 时直接返回当前顶层特征 `(B,C,H,W)`。否则初始化记忆列表、`num_obj_ptr_tokens`。

#### `_prepare_memory_conditioned_features`（第 650–705 行，收集记忆）

非初始帧：`select_closest_cond_frames` 选条件帧输出；再按 `memory_temporal_stride` 回溯 `num_maskmem-1` 个非条件帧；拼接 `maskmem_features` 与加 `maskmem_tpos_enc` 的位置编码（均为 `(S,B,C)` 序列格式）。

#### `_prepare_memory_conditioned_features`（第 707–765 行，对象指针）

若启用，从条件帧与非条件帧收集 `obj_ptr`，`get_1d_sine_pe` 时间编码，可能按 `mem_dim` 切分为多 token；append 到 memory 序列并更新 `num_obj_ptr_tokens`。

#### `_prepare_memory_conditioned_features`（第 766–791 行，注意力融合）

初始帧：`directly_add_no_mem_embed` 则加 `no_mem_embed` 后返回；否则 dummy `no_mem_embed` token。`memory = cat(...)`；`memory_attention(curr, memory, ...)` → `(HW,B,C)` reshape 为 `(B,C,H,W)`。

---

#### `_encode_new_memory`（第 793–834 行）

顶层 `pix_feat (B,C,H,W)`；eval 可非重叠约束掩码；点交互帧可二值化掩码；否则 sigmoid 及 scale/bias；`memory_encoder` → `maskmem_features` 与 pos enc；可选 `no_obj_embed_spatial` 叠加。

#### `_track_step`（第 836–865 行）

由高分辨 `vision_feats` 列表还原 `high_res_features` BCHW；若配置为 mask 直通则 `_use_mask_as_output`，否则 `_prepare_memory_conditioned_features` 得 `pix_feat`。

#### `_track_step`（第 866–892 行）

`prev_sam_mask_logits` 可作 `mask_inputs`；`_use_multimask` 决定是否多掩码；`_forward_sam_heads`。返回 `sam_outputs, high_res_features, pix_feat`。

#### `_encode_memory_in_output`（第 894–917 行）

`run_mem_encoder` 且 `num_maskmem>0` 时调用 `_encode_new_memory` 写入 `current_out`；否则 None。

#### `track_step`（第 919–997 行）

调用 `_track_step`；组装 `current_out`（pred_masks、high_res、obj_ptr；推理加 object_score_logits）；`_encode_memory_in_output`；返回 dict。

#### `_use_multimask`（第 999–1006 行）

根据点数与配置判断 SAM 头是否输出 3 掩码。

#### `_apply_non_overlapping_constraints`（第 1008–1024 行）

batch>1 时按像素取最大 logit 的实例保留，其余钳制为 ≤-10。

#### `set_binarize` / `set_imgsz`（第 1026–1038 行）

切换记忆编码二值化；更新 `image_size` 与 prompt 尺寸。

---

### `SAM3Model` 类（第 1041–1176 行）

#### `__init__`（第 1044–1137 行）

`super().__init__` 后把 `sam_mask_decoder` 换为带原版 `TwoWayTransformer` 的 `SAM2MaskDecoder`；默认 `image_size=1008`、`backbone_stride=14`。

#### `forward_image`（第 1139–1147 行）

调用 `image_encoder.forward_image_sam2`；高分辨路径同样预计算 conv_s0/s1。

#### `set_imgsz`（第 1149–1152 行）

`super().set_imgsz`；`memory_encoder.mask_downsampler.interpol_size = [s//14*16 for s in imgsz]`。

#### `_suppress_shrinked_masks`（第 1154–1164 行）

比较非重叠前后掩码面积，缩小过多则抑制 logits。

#### `_suppress_object_pw_area_shrinkage`（第 1166–1176 行）

非重叠约束后再按缩小比例过滤。