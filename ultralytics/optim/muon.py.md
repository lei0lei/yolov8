# `ultralytics/optim/muon.py`

本模块实现 Muon 优化相关核心：牛顿–舒尔茨五次迭代近似矩阵正交化、带 Nesterov 的动量 Muon 更新（含分桶批处理正交化），以及可与 SGD 混用的 `MuSGD` 优化器类。`zeropower_via_newtonschulz5` 在 bfloat16 下对 2D/3D 矩阵做归一化与可选转置后固定 5 步迭代。`muon_update` 用 `torch._foreach_*` 融合动量，再按列宽与尺度分桶、零填充行后批量调用正交化并写回各参数布局。`MuSGD` 继承 `torch.optim.Optimizer`：参数组 `use_muon=True` 时先施加 Muon 步再对独立 SGD 动量缓冲做经典动量更新，权重衰减仅作用于 SGD 分量。

## 逐段说明

### 函数 `zeropower_via_newtonschulz5`（第 10～50 行）

- 第 38 行：断言 `G` 为 2 维或 3 维。
- 第 39～40 行：reshape 为批矩阵并转 bfloat16，按最后一维范数加 `eps` 归一化。
- 第 41～42 行：若行数大于列数则转置，减少迭代成本。
- 第 43～47 行：固定系数 `a,b,c`，循环 5 次牛顿–舒尔z 式更新（`baddbmm` 组合 `A` 与 `A@A`）。
- 第 48～49 行：若曾转置则转回。
- 第 50 行：reshape 回 `G` 原始形状并返回。

### 函数 `muon_update`（第 53～127 行）

- 第 89～90 行：单张量时包装为长度为 1 的列表。
- 第 91～97 行：动量 `momentum *= beta` 后加 `(1-beta)*grad`；Nesterov 时 `updates = beta*momentum` 再加 `(1-beta)*grad`，否则 `updates = momentums`。
- 第 98～109 行：遍历每个 `update`，按 NCHW/NHWC 展平为 `(N, cols)` 矩阵，计算 `scale = sqrt(max(1, rows/cols))`，落入 `buckets` 键 `(cols, scale, device, dtype)`。
- 第 110～117 行：每桶内按行数降序，若相邻行数比超过 16× 则切分子组，限制填充浪费。
- 第 118～126 行：对每组零填充行矩阵 `X`，`foreach_add` 填入各行数据，批量 `zeropower_via_newtonschulz5`，转回梯度 dtype 并乘 `scale`；按原 stride 或 reshape 写回 `updates[i]`。
- 第 127 行：单张量返回首元素，否则返回列表。

### 类 `MuSGD`（第 130～264 行）

#### `MuSGD.__init__`（第 176～208 行）

- 第 199～205 行：`defaults` 收集 lr、momentum、weight_decay、nesterov、use_muon。
- 第 206 行：调用 `Optimizer.__init__`。
- 第 207～208 行：保存实例属性 `muon`、`sgd` 作为混合学习率缩放因子。

#### `MuSGD.step`（第 210～264 行）

- 第 229～232 行：可选 `closure` 在 `enable_grad` 下求 loss。
- 第 234～237 行：遍历参数组，过滤 `grad is not None`；空组跳过。
- 第 238～243 行：读取组内 lr、momentum、nesterov；对无 state 的参数初始化 `momentum_buffer`，Muon 组额外初始化 `momentum_buffer_SGD`。
- 第 244～253 行：`use_muon` 时调用 `muon_update`，`foreach_add` 将参数减去 `lr * self.muon * updates`；随后 SGD 缓冲列表指向 `momentum_buffer_SGD`，`lr *= self.sgd`。
- 第 254～255 行：非 Muon 组 SGD 缓冲为 `momentum_buffer`。
- 第 257～263 行：取 grads；`weight_decay != 0` 时 `grads += weight_decay * params`；`buffers *= momentum` 后 `buffers += grads`；Nesterov 时 `updates = grads + momentum * buffers`，否则 `updates = buffers`；`foreach_add` 参数减 `lr * updates`。
- 第 264 行：返回 `loss`。