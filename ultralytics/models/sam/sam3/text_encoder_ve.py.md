# `ultralytics/models/sam/sam3/text_encoder_ve.py`

CLIP 风格文本塔：`ResidualAttentionBlock`+`Transformer` 堆叠，`TextTransformer` 带因果掩码与多种池化，`VETextEncoder` 用 tokenizer 编码字符串并 `Linear` 投影到解码器 `d_model`。支持预编码元组输入以跳过重复 tokenize。

## 逐段说明

### `ResidualAttentionBlock`（行 17）

**`__init__`**：MHA batch_first；LayerNorm；可选 LayerScale；MLP 为 fc-gelu-proj。

**`attention`**：默认 k/v=q；非 bool mask 转 q.dtype。

**`forward`**：可选 `ln_1_kv` 处理 k/v；残差+LS+MLP 路径。

### `Transformer`（行 83）

堆叠 `layers` 个 block；训练+ckpt 时对每层 `checkpoint`；可 `torch.compile` forward。

### `text_global_pool`（行 132）

策略 `first`/`last`/`argmax`(需 text 张量找 EOT)/`none` 返回 pooled 与 tokens。

### `TextTransformer`（行 149）

嵌入+可学习位置；`build_causal_mask` 上三角 -inf；forward：token+pos → transformer → ln → pool → 线性或矩阵 `text_projection`；`output_tokens` 时返回 `(pooled, tokens)`。

### `VETextEncoder`（行 239）

内部 `TextTransformer(..., output_tokens=True)`；`resizer: Linear(width, d_model)`。

**`forward(text, input_boxes=None)`**（行 275）：
- **str 列表**：断言无 box；tokenizer→mask；`token_embedding` 与 encoder(tokenized) 得 memory；mask 反转 PyTorch 约定；memory 转 seq-first；`resizer`；`inputs_embeds` 转 seq-first 返回三元组。
- **已编码 tuple**：直接解包，禁止再传 box。