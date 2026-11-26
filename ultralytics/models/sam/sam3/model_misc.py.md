# `ultralytics/models/sam/sam3/model_misc.py`

SAM3 通用小模块：查询-文本点积打分、LayerScale、encoder-decoder 包装器、有效宽高比与正弦位置编码。供 decoder 参考点与 matcher 代价等处复用。

## 逐段说明

### `DotProductScoring`（行 16）

**`__init__`**：`prompt_mlp` 可选；`prompt_proj`/`hs_proj` 线性；`scale=1/sqrt(d_proj)`；`clamp_logits` 时保存 `clamp_max_val`。

**`mean_pool_text`**（行 39）：与 encoder 中逻辑类似，mask 为 padding=1，得 `(bs,d_model)` 池化向量。

**`forward(hs, prompt, prompt_mask)`**（行 50）：可选 MLP 处理 prompt；池化+双线性投影；`matmul` 得 scores×scale；可 `clamp_` 到 ±max。

### `LayerScale`（行 79）

`gamma` 参数；`forward` 原地或副本乘 gamma。

### `TransformerWrapper`（行 98）

保存 encoder/decoder、`num_queries`、`pos_enc_at_input_dec`；断言 `two_stage_type=="none"`；`_reset_parameters` 对多维参数 Xavier（排除 box/query/reference 名）。

### `get_valid_ratio(mask)`（行 130）

由 `(B,H,W)` padding mask 统计首列/首行非 pad 比例，堆叠 `(valid_w, valid_h)`。

### `gen_sineembed_for_position(pos_tensor, num_feats=256)`（行 141）

- 断言 `num_feats` 偶数，折半为频率维；
- 对 x/y（及可选 w/h）做 Transformer 经典 sin/cos 编码；
- 末维 2 或 4 坐标拼接输出。