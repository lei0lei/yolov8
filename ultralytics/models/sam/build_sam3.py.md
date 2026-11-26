# `ultralytics/models/sam/build_sam3.py`

本模块负责组装 SAM3 两类顶层模型：语义/open-vocabulary 图像模型 `SAM3SemanticModel`（`build_sam3_image_model`）与点/框提示及视频跟踪用的交互式 `SAM3Model`（`build_interactive_sam3`）。私有助手 `_create_vision_backbone` 固定 ViTDet + 双路 neck 视觉塔；`_create_sam3_transformer` 搭建检测用编码器-解码器 `TransformerWrapper`；`_load_checkpoint` 处理嵌套 `model` 键、SAM3.1 权重键名重命名，以及交互模式下 backbone/tracker 到 `SAM3Model` 字段的映射，并以 `strict=False` 加载。语义构建路径依赖 CLIP 分词器（缺失时通过 `check_requirements` 安装 Ultralytics CLIP fork）。交互构建复用视觉 backbone 可选、记忆 RoPE 注意力与 SAM2 风格跟踪超参，图像规格为 1008、步长 14。

## 逐段说明

### 导入（第 1–25 行）

- **第 7–10 行**：`MLP` 来自通用 `ultralytics.nn.modules.transformer`；`torch_load` 用于读 checkpoint。
- **第 12–25 行**：从 `modules` 与 `sam3` 子包拉取位置编码、记忆模块、`SAM3Model`、检测 transformer、几何编码、分割头、VL 骨干、ViT、文本编码器等组装类。

### 函数 `_create_vision_backbone`（第 28–71 行）

- **参数**：`compile_mode` 传给 ViT 与后续 pixel decoder；`enable_inst_interactivity` 控制 neck 是否 `add_sam2_neck`（实例交互分支）。
- **第 31–36 行**：`PositionEmbeddingSine`，`num_pos_feats=256`，`normalize=True`，`scale=None`，`temperature=10000`。
- **第 39–64 行**：`ViT` 骨干：`img_size=1008`，`pretrain_img_size=336`，`patch_size=14`，`embed_dim=1024`，`depth=32`，`heads=16`，`mlp_ratio=4.625`，`drop_path_rate=0.1`，绝对位置 + tile，`global_att_blocks=(7,15,23,31)`，`use_rope`/`use_interp_rope`，`window_size=24`，`retain_cls_token=False`，`ln_pre=True`，`compile_mode` 来自参数。
- **第 65–71 行**：`Sam3DualViTDetNeck` 包装：`d_model=256`，`scale_factors=[4,2,1,0.5]`，`trunk=vit_backbone`，`add_sam2_neck=enable_inst_interactivity`，注入上述 position encoding。

### 函数 `_create_sam3_transformer`（第 74–134 行）

#### 编码器 `TransformerEncoderFusion`（第 76–105 行）

- 单层 `TransformerEncoderLayer`：`d_model=256`，FFN 2048，`dropout=0.1`，`pre_norm=True`，自注意力与交叉注意力均为 8 头 `MultiheadAttention`，`batch_first=True`；位置编码仅在自注意力处启用（`pos_enc_at_attn=True`）。
- 外层：`num_layers=6`，`num_feature_levels=1`，`use_act_checkpoint=True`，`pool_text_with_mask=True`，`add_pooled_text_to_img_feat=False`，`frozen=False`。

#### 解码器 `TransformerDecoder`（第 106–132 行）

- 单层 `TransformerDecoderLayer`：交叉注意力 8 头，`use_text_cross_attention=True`。
- 外层：`num_queries=200`，`return_intermediate=True`，`box_refine=True`，`dac=True`，`boxRPB="log"`，`presence_token=True`，`use_act_checkpoint=True`，`num_o2m_queries=0`。

#### 返回（第 134 行）

`TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)`。

### 函数 `build_sam3_image_model`（第 137–257 行）

#### CLIP 依赖（第 149–155 行）

- `try import clip`；`ImportError` 时 `check_requirements("git+https://github.com/ultralytics/CLIP.git")` 后再导入。

#### 编译模式（第 157–158 行）

- `compile_mode = "default" if compile is True else compile or None`：`True` 等价字符串 `"default"`，`False` 为 `None` 不编译。

#### 视觉与文本（第 158–170 行）

- `vision_encoder = _create_vision_backbone(compile_mode, enable_inst_interactivity=True)`。
- `VETextEncoder`：`clip.simple_tokenizer.SimpleTokenizer()`，`d_model=256`，`width=1024`，`heads=16`，`layers=24`。

#### VL 骨干与检测头（第 170–173 行）

- `SAM3VLBackbone(visual=vision_encoder, text=text_encoder, scalp=1)`。
- `transformer = _create_sam3_transformer()`。

#### 点积打分（第 176–187 行）

- `DotProductScoring`：`d_model/d_proj=256`，`prompt_mlp` 为 2 层 residual `MLP` + `LayerNorm` 输出。

#### 分割头可选（第 190–212 行）

- **`enable_segmentation=True`**：`UniversalSegmentationHead`，`hidden_dim=256`，上采样 3 级，`pixel_decoder=PixelDecoder(..., compile_mode=compile_mode)`，带 8 头 cross attend prompt。
- **否则**：`segmentation_head = None`。

#### 几何序列编码器（第 215–239 行）

- `SequenceGeometryEncoder`：正弦位置编码，框直接投影/池化/位置编码，`d_model=256`，3 层 `TransformerEncoderLayer`（cross 时 keys 侧有 pos enc），`use_act_ckpt=True`，`add_cls=True`，`add_post_encode_proj=True`。

#### 组装、加载、eval（第 242–257 行）

- `SAM3SemanticModel`：`num_feature_levels=1`，`o2m_mask_predict=True`，`dot_prod_scoring`，`use_instance_query=False`，`multimask_output=True`，传入 backbone/transformer/geometry/segmentation。
- `_load_checkpoint(model, checkpoint_path)`（默认 `interactive=False`）。
- `model.eval()` 返回。

### 函数 `build_interactive_sam3`（第 260–352 行）

#### 记忆编码器（第 273 行）

- `MemoryEncoder(out_dim=64, interpol_size=[1152, 1152])`（比 SAM2 build 中默认插值尺寸更大）。

#### 记忆注意力（第 274–303 行）

- `MemoryAttention`：`batch_first=True`，`d_model=256`，`pos_enc_at_input=True`，`num_layers=4`。
- `MemoryAttentionLayer`：自注意力与交叉注意力均为 `RoPEAttention`（`embedding_dim=256`，`num_heads=1`，`feat_sizes=[72,72]`，交叉侧 `kv_in_dim=64`，`rope_k_repeat=True`），FFN 2048，`dropout=0.1`。

#### 骨干与编译（第 305–310 行）

- `compile_mode` 同上。
- **`with_backbone=True`**：`SAM3VLBackbone(scalp=1, visual=_create_vision_backbone(compile_mode), text=None)`。
- **否则**：`backbone = None`（仅加载跟踪头等场景的变体）。

#### `SAM3Model` 超参（第 311–345 行）

- `image_size=1008`，`backbone_stride=14`，`image_encoder=backbone`，记忆模块同上，`num_maskmem=7`，与 SAM2 类似的 sigmoid/多掩码/对象指针/跟踪标志；SAM3 交互路径下 `no_obj_embed_spatial`、`proj_tpos_enc_in_obj_ptrs`、`use_signed_tpos_enc_to_obj_ptrs` **恒为 True**（不依赖 checkpoint 名字）。
- `sam_mask_decoder_extra_args` 同 SAM2 动态多掩码稳定性配置。

#### 加载与返回（第 347–352 行）

- `_load_checkpoint(model, checkpoint_path, interactive=True)` 启用 tracker 键名 remap。
- `model.eval()` 返回。

### 函数 `_load_checkpoint`（第 355–394 行）

- **第 357–360 行**：读文件、`torch_load`；若有嵌套 `model` dict 则展开。
- **第 363–368 行**：全 state dict 键名替换：`tracker.model.interactive_` → `tracker.`，`interactivity_` 同理，`interactive_convs` → `sam2_convs`（兼容 SAM 3.1 嵌套命名）。
- **第 369 行**：`sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}`，先抽取 detector 前缀权重子集。
- **`interactive=True` 分支（第 370–392 行）**：
  - 将 `backbone.vision_backbone` 映射为 `image_encoder.vision_backbone`（在已过滤的 detector 键上 update）；
  - 从完整 `ckpt` 把 `tracker.transformer.encoder` → `memory_attention`，`tracker.maskmem_backbone` → `memory_encoder`；
  - 再把所有 `tracker.` 前缀去掉合并进 `sam3_image_ckpt`。
- **第 393–394 行**：`model.load_state_dict(sam3_image_ckpt, strict=False)`，允许交互模型缺少语义检测分支权重；返回 `model`。