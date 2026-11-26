# `ultralytics/models/sam/modules/utils.py`

本文件汇集 SAM / SAM2 骨干与注意力所需的**纯函数工具**：视频记忆帧选取、一维正弦位置编码、二维轴向 RoPE 复指数、窗口划分与相对位置偏置等。`memory_attention` 与 `blocks.RoPEAttention` 依赖 RoPE 相关函数；`encoders.Hiera` 与 `blocks` 中的 ViT/Hiera 块依赖窗口与相对位置；`sam.SAM2Model` 使用 `select_closest_cond_frames` 与 `get_1d_sine_pe` 组织跨帧记忆与对象指针时间编码。无 `nn.Module` 定义，可在 CPU/GPU 上作为张量运算辅助层使用。

## 逐段说明

### `select_closest_cond_frames`（第 14–63 行）

按当前 `frame_idx` 从 `cond_frame_outputs` 字典中最多选取 `max_cond_frame_num` 条：优先最近的前后各一帧，再按时间距离补足。`max_cond_frame_num==-1` 或字典较小时全选。返回 `(selected, unselected)` 两个字典。

### `get_1d_sine_pe`（第 66–89 行）

对 `pos_inds` 生成 `dim` 维正弦/余弦编码（Transformer 标准频率公式）。输出形状 `(*pos_inds.shape, dim)`。

### `init_t_xy`（第 92–118 行）

为 `end_x×end_y` 网格生成线性索引对应的 `(t_x, t_y)` 坐标（可按 `scale`、`offset` 缩放）。

### `compute_axial_cis`（第 121–151 行）

分别对 x/y 轴计算频率，外积得相位，再 `torch.polar` 合成为复数 RoPE 系数。输出 `(end_x*end_y, dim//2)` 复数视角下的最后一维（与 `apply_rotary_enc` 配合）。

### `reshape_for_broadcast`（第 154–174 行）

将 `freqs_cis` reshape 为可与 `xq` 末两维广播的形状；断言与 `x.shape[-2:]` 匹配。

### `apply_rotary_enc`（第 177–225 行）

将 `xq/xk` 末维拆为复数对，与 `freqs_cis` 相乘实现旋转；`torch.compile` 路径用实数运算避免复数融合问题。`repeat_freqs_k` 时在序列维重复频率以匹配更长的 key。空 key 序列时直接返回原 `xk`。

### `window_partition`（第 228–255 行）

输入 `(B,H,W,C)`，底部/右侧 pad 到 `window_size` 整除；重排为 `(B*n_win, window_size, window_size, C)`，返回 pad 后 `(Hp,Wp)`。

### `window_unpartition`（第 258–292 行）

逆操作还原 `(B,H,W,C)`，裁剪掉 pad 区域。

### `get_rel_pos`（第 295–332 行）

按 query/key 空间尺寸从可学习 `rel_pos (L,C)` 中查表或插值，得到 `(q_size, k_size, C)` 相对位置张量。

### `add_decomposed_rel_pos`（第 335–389 行）

MViTv2 式分解：用 `Rh、Rw` 与 `q` 的 einsum 得到高度/宽度注意力偏置，加到 `attn (B, qh*qw, kh*kw)` 上。

### `get_abs_pos`（第 392–453 行）

将 `(1, N, C)` 绝对位置（可选 cls token）reshape 为方阵，双三次插值或 tile 到目标 `(h,w)`，输出 `(1,h,w,C)` 或保留 cls 的 `(1,1+hw,C)`。

### `concat_rel_pos`（第 456–514 行）

将相对位置偏置通过扩展 q/k 通道拼入（替代显式加 attn bias），供 SDPA 路径使用；`rescale` 时调整 scale 以补偿维数变化。返回扩维后的 `q、k`。