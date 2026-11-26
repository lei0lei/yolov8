# `ultralytics/nn/modules/utils.py`

本文件收纳 `nn.modules` 包内复用的**纯函数与小工具**，不定义可插入 YAML 的层类。内容包括：深拷贝堆叠同一子模块（Transformer 解码层）、按先验概率初始化检测头偏置、在 `[0,1]` 区间上的逆 sigmoid（RT-DETR 等坐标与注意力采样）、以及**多尺度可变形注意力**的纯 PyTorch 实现（为 CoreML 等导出限制张量秩而改写）。这些函数被 `transformer.py`、`head` 或损失侧调用，保证注意力与初始化逻辑与模块实现同目录、避免循环导入。

## 逐段说明

### 函数 `_get_clones`（L14–L31）

- **输入**：模板模块 `module`，克隆份数 `n`。
- **行为**：`copy.deepcopy` 共 `n` 份，装入 `nn.ModuleList` 返回。
- **用途**：解码器中每层结构相同但参数独立。

### 函数 `bias_init_with_prob`（L34–L52）

- **输入**：正样本先验概率 `prior_prob`（默认 0.01）。
- **输出**：标量偏置 `−log((1−p)/p)`，即 logit 空间初值。
- **用途**：分类/对象性卷积 bias 初始化，使初始 sigmoid 输出约等于 `prior_prob`。

### 函数 `inverse_sigmoid`（L55–L77）

- **输入**：`x` 张量，先 `clamp` 到 `[0,1]`；`eps` 防止 `log(0)`。
- **计算**：`log(x₁/(1−x₂))`，其中 `x₁`、`(1−x)` 分别带下界 `eps`。
- **输出**：与 `x` 同形状的 logit 张量。

### 函数 `multi_scale_deformable_attn_pytorch`（L80–L127）

#### 张量约定

- `value`：`(bs, num_keys, num_heads, embed_dims)`，多尺度 key 在序列维拼接。
- `value_spatial_shapes`：各级 `(H, W)` 列表。
- `sampling_locations`：`(bs, num_queries, num_heads, num_levels*num_points, 2)`，归一化坐标。
- `attention_weights`：与采样点同级的权重。

#### 前向步骤

1. 由 `num_total_points // num_levels` 得每级采样点数 `num_points`。
2. `value` 重排为 `(bs*num_heads, embed_dims, num_keys)` 再按各级 `H*W` **split** 成 `value_list`。
3. 采样格点：`2 * sampling_locations - 1`（`grid_sample` 坐标系），按 level **split** 为 `sampling_grids`。
4. 对每层：`value_l`  reshape 为 `(bs*num_heads, embed_dims, h, w)`，`grid_sample` 双线性采样，结果入 `sampling_value_list`。
5. 权重 reshape 为 `(bs*num_heads, 1, num_queries, num_total_points)`。
6. 拼接各级采样值（通道维为 `embed_dims * num_total_points`），乘权重后在最后一维 **sum**，再 view 为 `(bs, num_heads*embed_dims, num_queries)` 并 `transpose` 为 `(bs, num_queries, num_heads*embed_dims)`。

**输出形状**：`(bs, num_queries, num_heads * embed_dims)`。