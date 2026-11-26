# `ultralytics/models/sam/modules/transformer.py`

本文件实现 SAM **掩码解码器核心的双向 Transformer**：图像 embedding 作为稠密 key/value，提示与输出 token 作为 query，经多层「token↔image」交叉注意力互相更新。`TwoWayTransformer` 堆叠 `TwoWayAttentionBlock` 并在末端再做一次 token→image 注意力；底层 `Attention` 提供可降维的多头注意力。SAM2 在 `blocks.py` 中通过 `SAM2TwoWayTransformer` 复用同一前向逻辑但替换注意力块类型；原版 SAM 与 SAM3 部分配置直接使用本文件的类。

## 逐段说明

### `TwoWayTransformer` 类（第 13–123 行）

#### `__init__`（第 40–79 行）

`depth` 层 `TwoWayAttentionBlock`（首层 `skip_first_layer_pe=True`）。`final_attn_token_to_image` 为末端 token 查询图像的 `Attention`（可 `downsample_rate=2`），`norm_final_attn` 收尾。

#### `forward`（第 81–123 行）

`image_embedding (B,C,H,W)` 与 `image_pe`：`flatten(2).permute(0,2,1)` → `(B, H*W, C)` 作为 `keys`。`queries = point_embedding (B, N_pts, C)`。每层 `layer(queries, keys, query_pe=point_embedding, key_pe=image_pe)` 双向更新。最后 `q = queries + point_embedding`，`k = keys + image_pe`，`attn_out = final_attn(q,k,v=keys)`，`queries = norm_final_attn(queries + attn_out)`。返回更新后的 `queries (B,N,C)`、`keys (B,HW,C)`（供解码头从 `keys` 恢复空间特征）。

---

### `TwoWayAttentionBlock` 类（第 126–238 行）

#### `__init__`（第 157–193 行）

四个子模块：token 自注意力、`cross_attn_token_to_image`（query 下采样率可为 2）、MLP、`cross_attn_image_to_token`（注意 forward 中 q/k 对调）。各阶段后 `LayerNorm`。

#### `forward`（第 195–238 行）

**(1) 自注意力**：首层可无 PE；否则 `q=k=queries+query_pe`，`v=queries`，残差 + `norm1`。

**(2) token→image**：`q=queries+query_pe`，`k=keys+key_pe`，`v=keys`，残差 + `norm2`。

**(3) MLP**：`queries += mlp(queries)`，`norm3`。

**(4) image→token**：`q=keys+key_pe`，`k=queries+query_pe`，`v=queries`（图像从 token 取信息），`keys` 残差 + `norm4`。`queries`、`keys` 形状不变。

---

### `Attention` 类（第 241–345 行）

#### `__init__`（第 271–299 行）

`internal_dim = embedding_dim // downsample_rate`；`q/k/v_proj` 将 Q 投到 `internal_dim`，K/V 可从 `kv_in_dim`（默认同 `embedding_dim`）投影。`out_proj` 投回 `embedding_dim`。

#### `_separate_heads` / `_recombine_heads`（第 301–313 行）

`(B,N,C) → (B, n_heads, N, C_head)` 与逆变换。

#### `forward`（第 315–345 行）

Q/K/V 投影后分头；`attn = softmax(q @ k^T / sqrt(d))`；`out = attn @ v`；合并头并 `out_proj`。输出 `(B, N_q, embedding_dim)`。