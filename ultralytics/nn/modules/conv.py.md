# `ultralytics/nn/modules/conv.py`

本文件实现 YOLO 家族中最常用的**卷积与特征路由算子**：标准卷积块、深度卷积、Ghost/Rep 结构、Focus 切片下采样、转置卷积上采样，以及 CBAM 通道/空间注意力。几乎所有 backbone、neck、head 的 YAML 层名（`Conv`、`C2f` 内部瓶颈等）最终都调用这里的 `Conv` 或其变体。`autopad` 保证 “same” 填充与 dilation 配套。推理阶段 `forward_fuse` 在 BN 折叠后与单卷积路径对齐。`Concat`/`Index` 服务于 FPN 拼接与多分支选取。整体上，该文件定义**空间尺寸与通道数的原子变换**，是 `block.py` 更大复合模块的积木。

## 逐段说明

### 函数 `autopad`（L30–L45）

根据 kernel `k`、可选 padding `p`、膨胀 `d` 计算有效核并默认 `p = k//2`，使输出空间尺寸与 stride 规则下的 “same” 对齐。

### 类 `Conv`（L48–L98）

#### `__init__`（L60–L76）

- `self.conv`：`nn.Conv2d(c1→c2, k, s, autopad, groups=g, dilation=d, bias=False)`。
- `self.bn`：`BatchNorm2d(c2)`。
- `self.act`：默认 `SiLU()`；`act=False` 用 `Identity`；或传入自定义 `nn.Module`。

#### `forward`（L78–L87）

`x: (B,c1,H,W)` → `conv` → `bn` → `act` → `(B,c2,H',W')`（`H',W'` 由 stride/kernel/padding 决定）。

#### `forward_fuse`（L89–L98）

仅 `conv` → `act`，无 BN（fuse 后使用）。

### 类 `Conv2`（L101–L149）

继承 `Conv`，额外 `cv2`: `1×1` 卷积 `c1→c2` 与主支 `k×k` **并行**相加后再 BN、激活。

#### `forward`（L127–L136）

`act(bn(conv(x) + cv2(x)))`，两路输出通道均为 `c2`，空间尺寸一致。

#### `fuse_convs`（L138–L149）

将 `cv2` 权重嵌入主卷积核中心位置相加，删除 `cv2`，`forward` 切回父类单卷积路径。

### 类 `LightConv`（L152–L185）

- `conv1`：`Conv(c1→c2, 1×1, act=False)`。
- `conv2`：`DWConv(c2→c2, k)`（深度卷积 + 激活）。
- `forward`：`x → (B,c2,H,W) → (B,c2,H',W')`。

### 类 `DWConv`（L188–L202）

继承 `Conv`，`groups=math.gcd(c1,c2)`，实现深度（或分组）卷积，通道规则同 `Conv`。

### 类 `DWConvTranspose2d`（L205–L219）

`nn.ConvTranspose2d` 子类，`groups=gcd(c1,c2)`，无 BN/激活包装。

### 类 `ConvTranspose`（L222–L271）

- `conv_transpose`：`ConvTranspose2d(c1→c2, k, s, p, bias=not bn)`。
- 可选 `bn` 与 `act`（默认 SiLU）。
- `forward` / `forward_fuse`：转置卷积 →（BN）→ 激活，通常用于上采样（如 `Proto`）。

### 类 `Focus`（L274–L311）

- `conv`：`Conv(c1*4 → c2, k, s, …)`。
- `forward`：将 `x` 按步长 2 在 H、W 上取四个子格 **concat** 到通道维得 `(B,4*c1,H/2,W/2)`，再卷积到 `(B,c2,…)`。

### 类 `GhostConv`（L314–L353）

- `c_ = c2//2`；`cv1`: `c1→c_`；`cv2`: 在 `c_` 上做 cheap `5×5` 深度分组卷积 `c_→c_`。
- `forward`：`y=cv1(x)`，`cat(y, cv2(y))` → `(B,c2,H,W)`。

### 类 `RepConv`（L356–L511）

训练时三分支：`conv1` 3×3、`conv2` 1×1、可选 `bn` 恒等（`c1==c2,s==1`）。

#### `forward`（L412–L422）

`act(conv1(x)+conv2(x)+bn(x))`。

#### `get_equivalent_kernel_bias` / `_fuse_bn_tensor` / `_pad_1x1_to_3x3_tensor`（L424–L485）

将各分支 BN 并入权重，1×1 pad 到 3×3，求和得等效单核与 bias。

#### `fuse_convs`（L487–L511）

生成带 bias 的单一 `nn.Conv2d`，删训练分支，推理走 `forward_fuse`。

### 类 `ChannelAttention`（L514–L548）

- `pool`: GAP → `(B,C,1,1)`；`fc`: `Conv2d(C→C,1×1)`；`act`: Sigmoid。
- `forward`：`x * sigmoid(fc(pool(x)))`，形状不变。

### 类 `SpatialAttention`（L551–L582）

- `cv1`: `2→1` 通道（由 mean/max over C 拼接）卷积，`kernel_size` 3 或 7。
- `forward`：`x * sigmoid(cv1(cat(mean,max)))`。

### 类 `CBAM`（L585–L615）

串联 `ChannelAttention(c1)` → `SpatialAttention(kernel_size)`。

### 类 `Concat`（L618–L643）

沿维度 `d`（默认 1，通道维）对张量列表 `cat`。

### 类 `Index`（L646–L671）

从列表输入 `x[self.index]` 取单张特征图，用于多输出模块选一。