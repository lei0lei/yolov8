# `ultralytics/models/sam/sam3/vitdet.py`

SAM3 视觉主干：Detectron2 ViTDet 风格 `ViT`，块内 `Attention` 支持相对位置与 2D RoPE，`Block` 可选窗口注意力，`forward` 在全局注意力 block 导出 NCHW 特征列表。依赖 `timm` DropPath/Mlp 与 SAM 模块中的 patch/window/rope 工具。

## 逐段说明

### `Attention`（行 40）

**`__init__`**：QKV 与 proj 线性；`use_rel_pos` 时初始化 rel_pos_h/w 与 `relative_coords`；`use_rope` 时 `compute_axial_cis` 预计算 `freqs_cis`，cls 时可拼零旋转。

**`_setup_rel_pos` / `_setup_rope_freqs`**：按输入尺寸重建参数（`set_imgsz` 会重调）。

**`_apply_rope`**：调用 `apply_rotary_enc`。

**`forward(x)`**（行 167）：
- 支持 `(B,H,W,C)` 或 `(B,L,C)`；reshape qkv → multi-head；
- rope 与 rel_pos（`concat_rel_pos`）后 SDPA；
- 还原空间维并 `proj`。

### `Block`（行 216）

**`__init__`**：`check_requirements("timm")`；Attention 的 `input_size` 在窗口块为 window 大小；DropPath、LayerScale、MLP。

**`forward`**（行 295）：norm1 → 可选 window_partition → attn+LS → unpartition → 残差；norm2+MLP+LS 残差。

### `ViT`（行 315）

**`__init__`（要点）**：
- `window_block_indexes` = 非 global 层；`rel_pos_blocks` 按索引或全真；
- `retain_cls_token` 时断言无窗口/rel_pos，学习 `class_embedding`；
- `PatchEmbed`；可选绝对位置 `pos_embed` 与 tile；
- 深度 `dpr` 线性 schedule；每层 `Block` 窗口/global 切换；
- `return_interm_layers` 控制 `channel_list`；可 compile 与 act ckpt。

**`_init_weights`**：Linear trunc_normal；LayerNorm 偏置 0 权重 1。

**`forward(x)`**（行 503）：
- patch_embed；
- 可选 cls 拼序列；
- `get_abs_pos` 加位置；`ln_pre`；
- 逐 block：训练 ckpt 或直连；在 `full_attn_ids` 末或中间层收集特征：去 cls、`ln_post`（末层）、reshape 为 BCHW append；
- 返回 list[Tensor]。

**`set_imgsz`**（行 548）：仅对 `window_size==0` 的块重设 rel_pos 与 rope 频率，尺寸为 `imgsz//patch_size`。