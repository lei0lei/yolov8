# `ultralytics/models/sam/build.py`

本文件源自 Meta SAM/SAM2 构建逻辑，经 Ultralytics 适配后负责按规格实例化 `SAMModel`（SAM1、Mobile-SAM）与 `SAM2Model`（SAM2 / SAM2.1），并从本地或远程 checkpoint 载入权重。核心私有函数 `_build_sam`、`_build_sam2` 拼装图像编码器、提示编码器、掩码解码器及（SAM2）记忆注意力与记忆编码器；公开 `build_sam_vit_*`、`build_mobile_sam`、`build_sam2_*` 仅传入各变体的 ViT/Hiera/TinyViT 超参。`_load_checkpoint` 统一下载、`torch_load` 与可选的嵌套 `model` 键解包。顶层 `build_sam(ckpt)` 根据 checkpoint 文件名后缀在 `sam_model_map` 中匹配 builder，找不到则 `FileNotFoundError`。构建完成后模型一律 `eval()` 再返回。

## 逐段说明

### 版权、导入与依赖（第 1–21 行）

- **第 9 行**：`partial` 用于给 `LayerNorm` 固定 `eps=1e-6`。
- **第 13–14 行**：`attempt_download_asset` 在 checkpoint 为资产名时拉取权重；`torch_load` 为安全加载封装。
- **第 16–21 行**：从 `.modules` 导入解码器、各类编码器、记忆模块、`SAMModel`/`SAM2Model`、`TinyViT`、`TwoWayTransformer`，构成装配所需的全部子模块类。

### 函数 `_load_checkpoint`（第 24–36 行）

- **第 26–27 行**：`checkpoint is None` 时原样返回 `model`，不读盘。
- **第 29 行**：`attempt_download_asset(checkpoint)` 将 URL/短名解析为本地路径。
- **第 30–31 行**：二进制打开文件，`torch_load(..., weights_only=True)` 得到 `state_dict`。
- **第 33–34 行**：若顶层含 `model` 且其值为 `dict`，则把内层当作真正的 state dict（兼容 Ultralytics 式打包格式）。
- **第 35 行**：`model.load_state_dict(state_dict)` 严格加载后返回 `model`。

### 函数 `build_sam_vit_h`（第 39–47 行）

调用 `_build_sam`，编码器参数：`embed_dim=1280`，`depth=32`，`heads=16`，全局注意力层索引 `[7,15,23,31]`，传入 `checkpoint`。对应资产名 `sam_h.pt`。

### 函数 `build_sam_vit_l`（第 50–58 行）

`_build_sam`：`embed_dim=1024`，`depth=24`，`heads=16`，全局注意力 `[5,11,17,23]`。对应 `sam_l.pt`。

### 函数 `build_sam_vit_b`（第 61–69 行）

`_build_sam`：`embed_dim=768`，`depth=12`，`heads=12`，全局注意力 `[2,5,8,11]`。对应 `sam_b.pt`（默认规格）。

### 函数 `build_mobile_sam`（第 72–81 行）

`_build_sam` 且 `mobile_sam=True`：编码器维度/深度/头数为列表（TinyViT 各 stage），`encoder_global_attn_indexes=None`，走 `TinyViT` 分支而非 `ImageEncoderViT`。对应 `mobile_sam.pt`。

### 函数 `build_sam2_t`（第 84–94 行）

`_build_sam2`：Hiera `embed_dim=96`，`stages=[1,2,7,2]`，`num_heads=1`，全局块 `[5,7,9]`，窗口规格 `[8,4,14,7]`，骨干通道 `[768,384,192,96]`。对应 `sam2_t.pt` / `sam2.1_t.pt`。

### 函数 `build_sam2_s`（第 97–107 行）

`_build_sam2`：`stages=[1,2,11,2]`，全局块 `[7,10,13]`，其余 tiny 同结构。对应 `sam2_s.pt` / `sam2.1_s.pt`。

### 函数 `build_sam2_b`（第 110–121 行）

`_build_sam2`：`embed_dim=112`，`stages=[2,3,16,3]`，`num_heads=2`，全局块 `[12,16,20]`，`encoder_window_spatial_size=[14,14]`，通道 `[896,448,224,112]`。对应 `sam2_b.pt` / `sam2.1_b.pt`。

### 函数 `build_sam2_l`（第 124–134 行）

`_build_sam2`：`embed_dim=144`，`stages=[2,6,36,4]`，全局块 `[23,33,43]`，窗口 `[8,4,16,8]`，通道 `[1152,576,288,144]`。对应 `sam2_l.pt` / `sam2.1_l.pt`。

### 函数 `_build_sam`（第 137–225 行）

#### 超参与嵌入尺寸（第 162–165 行）

- `prompt_embed_dim=256`，`image_size=1024`，`vit_patch_size=16`，`image_embedding_size=64`（1024/16）。

#### 图像编码器分支（第 166–198 行）

- **`mobile_sam=True`（第 167–182 行）**：实例化 `TinyViT`，`img_size=1024`，`embed_dims/depths/num_heads` 来自入参列表，`window_sizes=[7,7,14,7]`，`mlp_ratio=4`，无 dropout/drop path，`mbconv_expand_ratio=4`，`local_conv_size=3`，`layer_lr_decay=0.8`。
- **否则（第 184–197 行）**：`ImageEncoderViT`，`depth/embed_dim/num_heads` 为标量入参，`patch_size=16`，`qkv_bias=True`，`use_rel_pos=True`，`global_attn_indexes` 来自入参，`window_size=14`，`out_chans=256`（与 prompt 维一致），`norm_layer=partial(LayerNorm, eps=1e-6)`。

#### SAM1 三件套与归一化（第 199–221 行）

- **`SAMModel`（第 199–221 行）**：
  - `image_encoder`：上一步产物；
  - `PromptEncoder`：`embed_dim=256`，特征图 spatial `(64,64)`，输入图 `(1024,1024)`，`mask_in_chans=16`；
  - `MaskDecoder`：`num_multimask_outputs=3`，内嵌 `TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)`，`iou_head_depth=3`，`iou_head_hidden_dim=256`；
  - `pixel_mean/std`：ImageNet 风格 BGR 均值方差（与原始 SAM 一致）。

#### 权重与 eval（第 222–225 行）

- `checkpoint is not None` 时 `_load_checkpoint(sam, checkpoint)`；
- `sam.eval()` 后返回。

### 函数 `_build_sam2`（第 228–316 行）

#### 默认形参与文档（第 228–256 行）

函数签名带 SAM2-large 量级默认值；各 `build_sam2_*` 覆盖为对应 tiny/small/base/large 配置。

#### 图像编码器 `ImageEncoder`（第 257–273 行）

- **`trunk=Hiera`**：`embed_dim`、`num_heads`、`stages`、`global_att_blocks`、`window_pos_embed_bkg_spatial_size`（默认 `(7,7)`，base 可为 `(14,14)`）、`window_spec` 均来自入参。
- **`neck=FpnNeck`**：`d_model=256`，`backbone_channel_list` 来自入参，`fpn_top_down_levels=[2,3]`，插值 `nearest`。
- **`scalp=1`**：丢弃最高分辨率 FPN 级（SAM2 标准设定）。

#### 记忆模块（第 274–275 行）

- `MemoryAttention`：`d_model=256`，`pos_enc_at_input=True`，`num_layers=4`，单层配置为默认 `MemoryAttentionLayer()`。
- `MemoryEncoder`：`out_dim=64`。

#### SAM2.1 标志（第 277 行）

- `is_sam2_1 = checkpoint is not None and "sam2.1" in checkpoint`：路径字符串含 `sam2.1` 时启用额外对象指针时空嵌入相关开关。

#### `SAM2Model` 构造（第 278–311 行）

传入 `image_encoder`、`memory_attention`、`memory_encoder`，`num_maskmem=7`，`image_size=1024`，记忆编码 sigmoid 缩放/偏置 `20.0` / `-10.0`，以及一系列跟踪与多掩码行为布尔量（如 `use_high_res_features_in_sam`、`multimask_output_for_tracking`、`pred_obj_scores` 等）。
- **SAM2.1 专有（第 303–305 行）**：当 `is_sam2_1` 时 `no_obj_embed_spatial`、`proj_tpos_enc_in_obj_ptrs`、`use_signed_tpos_enc_to_obj_ptrs` 为 `True`。
- **`sam_mask_decoder_extra_args`（第 306–310 行）**：动态多掩码稳定性阈值 `delta=0.05`、`thresh=0.98`。

#### 权重与 eval（第 313–316 行）

- 有 `checkpoint` 则 `_load_checkpoint`；
- `sam2.eval()` 返回。

### 字典 `sam_model_map`（第 319–332 行）

键为官方权重文件名（含 SAM2.1 与 SAM2 共享同一 builder 的映射，如 `sam2.1_t.pt` → `build_sam2_t`），值为无参调用时再把 `ckpt` 路径传回 `_load_checkpoint` 的工厂函数。

### 函数 `build_sam`（第 335–366 行）

- **第 357–361 行**：`model_builder` 初始 `None`；`ckpt = str(ckpt)` 兼容 `Path`；遍历 `sam_model_map` 键，若 `ckpt.endswith(k)` 则选中对应 builder（**后缀匹配**，故完整路径 `.../sam_b.pt` 可匹配）。
- **第 363–364 行**：未匹配则 `FileNotFoundError`，错误信息列出 `sam_model_map.keys()`。
- **第 366 行**：`return model_builder(ckpt)`，builder 内部用同一 `ckpt` 路径构建并加载权重。