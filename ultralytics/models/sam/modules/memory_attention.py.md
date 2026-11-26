# `ultralytics/models/sam/modules/memory_attention.py`

本模块实现 SAM2 视频分割中的**记忆注意力栈**：当前帧的视觉 token 经自注意力后，与历史帧编码得到的 memory token（及可选的对象指针 token）做交叉注意力，再经 FFN，使当前帧特征带上时序与对象上下文。`MemoryAttentionLayer` 为单层（RoPE 自注意力 + RoPE 交叉注意力 + MLP）；`MemoryAttention` 堆叠多层并处理 batch-first / seq-first 维度的转置。该模块位于 `SAM2Model` 的 backbone 特征与 SAM 解码头之间，与 `encoders.MemoryEncoder` 写入的 `maskmem_features` 配套使用。

## 逐段说明

### `MemoryAttentionLayer` 类（第 13–169 行）

#### `__init__`（第 54–107 行）

构造默认 `RoPEAttention` 自注意力（`embedding_dim=256`，单头）与交叉注意力（`kv_in_dim=64`，`rope_k_repeat=True` 以适配 memory 序列长于当前帧空间 token）。配置三处 `LayerNorm`、FFN（`linear1`→ReLU→dropout→`linear2`）及位置编码注入标志：`pos_enc_at_attn`、`pos_enc_at_cross_attn_queries`、`pos_enc_at_cross_attn_keys`（默认仅给 cross-attn 的 key 加 `pos`）。

#### `_forward_sa`（第 109–115 行）

输入 `tgt` 形状 `(N, L, D)`。`norm1` 后对 `q=k` 可选加 `query_pos`，`self_attn` 输出与 `tgt` 残差相加并经 `dropout1`。输出仍为 `(N, L, D)`。

#### `_forward_ca`（第 117–140 行）

`norm2(tgt)` 作为 query（可选 `+query_pos`）；`memory` 作 key/value（key 可选 `+pos`）。`num_k_exclude_rope>0` 时传给 `RoPEAttention`，末尾若干 key 不参与 RoPE（对象指针 token）。交叉注意力结果残差加回 `tgt`。形状保持 `(N, L, D)`。

#### `forward`（第 142–169 行）

依次：`_forward_sa` → `_forward_ca` → `norm3` → FFN（`D→dim_feedforward→D`）残差。返回 `(N, L, D)`。

---

### `MemoryAttention` 类（第 172–303 行）

#### `__init__`（第 203–230 行）

`copy.deepcopy(layer)` 复制 `num_layers` 份 `MemoryAttentionLayer`；`norm` 为输出 `LayerNorm`；`pos_enc_at_input` 控制是否在栈入口对 `curr` 加 `0.1 * curr_pos`；`batch_first` 控制内部是否 `(L,B,D)↔(B,L,D)` 转置。

#### `forward`（第 232–270 行，入口与张量整理）

若 `curr`/`curr_pos` 为单元素 list，解包为张量。断言 `curr` 与 `memory` 的 batch 维一致。`output = curr`；若 `pos_enc_at_input` 则 `output += 0.1 * curr_pos`。`batch_first` 时对 `output`、`curr_pos`、`memory`、`memory_pos` 做 `transpose(0,1)`：`(L,B,*) → (B,L,*)`。

#### `forward`（第 271–303 行，层循环与输出）

每层调用 `MemoryAttentionLayer`，对 `RoPEAttention` 交叉注意力传入 `num_k_exclude_rope=num_obj_ptr_tokens`。栈末 `LayerNorm`。若 `batch_first`，再 `transpose` 回 `(L, B, D)`。输出与输入 `curr` 同形状 `(L, B, d_model)`。