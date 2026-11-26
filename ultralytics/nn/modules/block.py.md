# `ultralytics/nn/modules/block.py`

本文件是 Ultralytics YOLO **骨干与颈部的主体积木库**：CSP 瓶颈族（`C2f`、`C3`、`C3k2`）、池化金字塔（`SPP`/`SPPF`）、下采样（`SCDown`、`ADown`）、Rep/Ghost/HGNet 变体、自注意力与区域注意力（`PSA`、`A2C2f`）、视觉语言对比头、掩码原型 `Proto`、边框分布积分 `DFL`，以及 Flow 姿态用的 `RealNVP`。各层通过 1×1 分裂/拼接通道、瓶颈堆叠与可选 shortcut，在保持多尺度特征图 `(B,C,H,W)` 流动的同时调节感受野与算力。许多类专为特定代际 YOLO（v8/v9/v10/v11/v26、YOLOE、World）服务，由 YAML 按符号名实例化。理解本文件即可掌握**特征从 backbone 到 neck 再喂给 head 之前的几乎全部结构化变换**。

## 逐段说明

### 类 `DFL`（L60–L82）

#### `__init__`（L66–L76）

- 固定 `nn.Conv2d(c1, 1, 1, bias=False)`，`weight` 初始化为 `[0,1,…,c1−1]`（每侧边 `reg_max` 个 bin）。
- `c1` 默认 16，对应 DFL 分布细分数。

#### `forward`（L78–L81）

- 输入 `x`: `(B, 4*c1, A)`（A 为 anchor 数）。
- reshape 为 `(B, 4, c1, A)`，`softmax` 在 bin 维，`1×1 conv` 求期望 → `(B, 4, A)` 四边距离解码。

### 类 `Proto`（L85–L104）

- `cv1`: `c1→c_` 3×3；`upsample` 转置卷积 2×上采样；`cv2` 3×3；`cv3` 到 `c2`（掩码原型数）。
- `forward`: 特征 → 上采样 ×2 → `(B, c2, 2H, 2W)`。

### 类 `HGStem`（L107–L140）

- `stem1` `c1→cm` stride 2；`stem2a/b` 在 pad 后分两支；`pool` 对 stem1 输出；`cat` 通道加倍 → `stem3` stride 2 → `stem4` 到 `c2`。
- 输出空间约为输入 1/4，通道 `c2`。

### 类 `HGBlock`（L143–L185）

- `m`: `n` 个 `LightConv` 或 `Conv`，首块 `c1→cm` 后继 `cm→cm`。
- `sc`: 压缩 `cat(x,各层输出)` 通道 `c1+n*cm → c2//2`；`ec`: 扩展到 `c2`。
- `shortcut` 当 `c1==c2` 时残差加输入。

### 类 `SPP`（L188–L208）

- `cv1` 减半通道到 `c_`；并行 `len(k)` 个 `MaxPool2d`（stride 1）与原始支路 `cat` → `cv2` 到 `c2`。

### 类 `SPPF`（L211–L251）

- `cv1` 到 `c_`（无激活）；循环 `n` 次同核 `MaxPool2d`（CPU 可用可分离池化近似）拼接 `n+1` 路 → `cv2`。
- 可选 `shortcut`：`c1==c2` 时加回输入。

### 类 `C1`（L254–L272）

- `cv1` `c1→c2`；`m` 为 `n` 个 `3×3 Conv(c2→c2)`。
- `forward`: `y=cv1(x)`，`m(y)+y`。

### 类 `C2`（L275–L299）

- 隐藏宽 `c=int(c2*e)`；`cv1` `c1→2c`；`m` 为 `n` 个 `Bottleneck(c,c)`；`cv2` 将 `cat(m(a), b)` 压到 `c2`（`a,b` 为 chunk 两半）。

### 类 `C2f`（L302–L333）

- 同 C2 分裂，但瓶颈输出**级联拼接**所有中间特征（共 `2+n` 段 × `c` 通道）再 `cv2`。
- `forward_split` 用 `split` 替代 `chunk`，行为等价。

### 类 `C3`（L336–L359）

- 双 1×1：`cv1/cv2` 均 `c1→c_`；一支过 `n` 个 `Bottleneck`，另一支直连；`cv3` `2*c_→c2`。

### 类 `C3x`（L362–L378）

- 继承 `C3`，瓶颈内核改为 `(1,3)` 与 `(3,1)` 十字卷积。

### 类 `RepC3`（L381–L402）

- 两路 1×1 到 `c_`，一路 `RepConv` 堆栈，一路捷径；相加后可选 `cv3` `c_→c2`。

### 类 `C3TR`（L405–L421）

- 继承 `C3`，`m` 换为 `TransformerBlock(c_, c_, 4, n)`（含多头注意力）。

### 类 `C3Ghost`（L424–L440）

- 继承 `C3`，`m` 为 `n` 个 `GhostBottleneck`。

### 类 `GhostBottleneck`（L443–L468）

- `conv`: `GhostConv` →（stride2 时 `DWConv`）→ `GhostConv` 无激活。
- `shortcut`: stride2 时 `DWConv+1×1` 对齐尺寸与通道。
- 输出与 shortcut 相加。

### 类 `Bottleneck`（L471–L493）

- `cv1` `c1→c_`（`k[0]`），`cv2` `c_→c2`（`k[1]`, groups `g`）。
- `shortcut` 且 `c1==c2` 时残差。

### 类 `BottleneckCSP`（L496–L524）

- 四卷积 CSP：`cv1` 进瓶颈链，`cv2` 旁路，`cv3` 旁路后 `cat`，`bn+SiLU`，`cv4` 出 `c2`。

### 类 `ResNetBlock`（L527–L548）

- 瓶颈 `c1→c2→c2→e*c2`（`e` 默认 4）；shortcut `1×1` 当 stride≠1 或通道不等。
- `ReLU(主支+shortcut)`。

### 类 `ResNetLayer`（L551–L579）

- `is_first`: `7×7 stride2 + MaxPool`；否则 `n` 个 `ResNetBlock` 堆叠（首块可 stride）。
- 输出通道逻辑见 docstring（首层 `c2`，否则有效 `e*c2`）。

### 类 `MaxSigmoidAttnBlock`（L582–L630）

#### `__init__`（L585–L603）

- 每头维 `hc=c2//nh`；可选 `ec` 将 `c1` 投到 `ec`；`gl` 将 guide `gc→ec`；`proj_conv` `c1→c2` 3×3。
- 可学习 `scale`、`bias(nh)`。

#### `forward`（L605–L630）

- `guide` 线性后 reshape 为 `(bs, *, nh, hc)`；`embed` 来自 `x`（或经 `ec`）`(bs, nh, hc, h, w)`。
- 注意力权重 `aw`：embed 与 guide 矩阵乘、max、缩放、加 bias、sigmoid。
- `proj_conv(x)` 按头 reshape，乘 `aw`，展平回 `(bs, c2, h, w)`。

### 类 `C2fAttn`（L633–L696）

- 结构同 `C2f`，但 `cv2` 输入通道 `(3+n)*c`（多一支注意力输出）；`attn` 为 `MaxSigmoidAttnBlock`。
- `forward(x, guide)`：级联瓶颈后在末路特征上跑 `attn`，再 concat 压回 `c2`。

### 类 `ImagePoolingAttn`（L699–L761）

#### `__init__`（L702–L729）

- 多尺度 `ch` 各配 `1×1 Conv→ec` 与 `AdaptiveMaxPool2d(k,k)`；`query/key/value/proj` 为 LayerNorm+Linear（`ct` 文本维 ↔ `ec`）。

#### `forward`（L731–L761）

- 各层特征池化到 `k×k` patch，拼成 key/value 序列；`text` 作 query 多头注意力；残差 `proj(…) * scale + text`，增强文本嵌入。

### 类 `ContrastiveHead`（L764–L787）

- 可学习 `logit_scale`、`bias`；L2 归一化图像与文本，批量矩阵乘得区域-类相似度图，再缩放加偏置。

### 类 `BNContrastiveHead`（L790–L836）

- 图像侧 `BatchNorm2d` 替代 L2；文本仍 L2 归一；`fuse()` 后 `forward_fuse` 直接返回 `x`。

### 类 `RepBottleneck`（L839–L855）

- 继承 `Bottleneck`，`cv1` 换 `RepConv`。

### 类 `RepCSP`（L858–L874）

- 继承 `C3`，瓶颈换 `RepBottleneck`。

### 类 `RepNCSPELAN4`（L877–L907）

- `cv1` 到 `c3` 两半；`cv2/cv3` 各为 `RepCSP+3×3` 链式；`cv4` 融合 `c3+2*c4 → c2`。

### 类 `ELAN1`（L910–L927）

- 覆写 `RepNCSPELAN4` 子模块为普通 `Conv` 链（无 RepCSP），接口参数同父类。

### 类 `AConv`（L930–L946）

- 先 `avg_pool2d` 2×，再 `Conv` stride2 3×3，`c1→c2`，空间约半。

### 类 `ADown`（L949–L971）

- avg pool 后通道 **chunk 两半**：一半 `cv1` 3×3 s2，一半 maxpool + `cv2` 1×1；`cat` 得 `c2`。

### 类 `SPPELAN`（L974–L998）

- 类似 SPPF：1×1 降 `c3`，三次串联 MaxPool，4 路 concat → 1×1 到 `c2`。

### 类 `CBLinear`（L1001–L1021）

- 单卷积输出 `sum(c2s)` 通道，`forward` 按 `c2s` **split** 为多 tensor 元组。

### 类 `CBFuse`（L1024–L1048）

- 输入列表：前几项为 CBLinear 多输出中的选定索引，末项为目标特征图。
- 将前几路 `nearest` 插值到末项空间尺寸，与末项 **stack sum** 融合。

### 类 `C3f`（L1051–L1076）

- 双 1×1 起步（`cv2(x)` 与 `cv1(x)` 并列），级联瓶颈，拼 `(2+n)*c_` → `cv3`。

### 类 `C3k2`（L1079–L1116）

- 继承 `C2f`，内层可为：`Bottleneck+PSABlock`、`C3k` 或纯 `Bottleneck`（由 `attn`/`c3k` 控制）。

### 类 `C3k`（L1119–L1137）

- 继承 `C3`，瓶颈 `k×k` 同尺寸卷积（默认 k=3）。

### 类 `RepVGGDW`（L1140–L1202）

- 并联 `7×7 DWConv` 与 `3×3 DWConv`（无激活），相加后 `SiLU`。
- `fuse()` 将两深度卷积 BN 合并为单 `Conv` 权重。

### 类 `CIB`（L1205–L1247）

- 紧凑倒置块：`DW 3×3(c1)` → `1×1→2*c_` → `RepVGGDW` 或 `DW 3×3(2*c_)` → `1×1→c2` → `DW 3×3(c2)`。
- 可选残差 `c1==c2`。

### 类 `C2fCIB`（L1250–L1278）

- `C2f` 的 `m` 换为 `n` 个 `CIB`。

### 类 `Attention`（L1281–L1344）

#### `__init__`（L1302–L1319）

- 头维 `head_dim=dim//num_heads`，key 维 `key_dim=attn_ratio*head_dim`。
- `qkv`: `Conv(dim → dim+2*nh*key_dim, 1)`；`proj`/`pe`：`1×1` 与 `3×3 DW`。

#### `forward`（L1321–L1344）

- `N=H*W`，qkv reshape 分 q/k/v；标准 `(q@k)softmax@v` 或 CoreML 的 `scaled_dot_product_attention`；加 `pe(v)`，再 `proj`。

### 类 `PSABlock`（L1347–L1394）

- `Attention(c)` + 两 1×1 FFN（扩 2c 再压回）；可选残差绕注意力与 FFN。

### 类 `PSA`（L1397–L1449）

- 要求 `c1==c2`；`cv1` 到 `2c`，split 为旁路 `a` 与支路 `b`；`b` 上 Attention+FFN 残差；`cv2` cat 回 `c1`。

### 类 `C2PSA`（L1452–L1504）

- 同 PSA split，但 `b` 过 `n` 个串联 `PSABlock`。

### 类 `C2fPSA`（L1507–L1543）

- `C2f` 级联结构，`m` 为 `PSABlock` 列表；要求 `c1==c2`。

### 类 `SCDown`（L1546–L1591）

- `cv1` `c1→c2` 1×1；`cv2` 深度卷积 `c2→c2` stride `s` 完成下采样。

### 类 `TorchVision`（L1594–L1657）

#### `__init__`（L1613–L1640）

- 动态 `import torchvision`，`get_model` 加载权重；可选 unwrap 截断尾部 `truncate` 层为 `nn.Sequential`。
- 不 unwrap 时 `head/heads` 置 `Identity`。

#### `forward`（L1642–L1657）

- `split=True` 时返回 `[x, layer1(x), …]` 链；否则整段 sequential 输出。

### 类 `AAttn`（L1660–L1746）

#### `__init__`（L1685–L1702）

- `qkv`: `Conv(dim → 3*all_head_dim, 1)`；`proj`；`pe` 为 `7×7` 分组卷积。

#### `forward`（L1710–L1746）

- 可选 `area>1` 将序列分块做注意力再还原形状；标准 qkᵀ softmax v；输出加 `pe(v)` 后 `proj` → `(B,dim,H,W)`。

### 类 `ABlock`（L1749–L1811）

- `AAttn` + 两 1×1 MLP（`mlp_ratio` 控制隐层）；双残差。

### 类 `A2C2f`（L1814–L1893）

- `cv1` `c1→c_`；`m` 为 `n` 段，每段 either **2×ABlock**（`a2=True`）或 `C3k`。
- 级联 concat → `cv2` 到 `c2`；`a2+residual` 时用可学习 `gamma` 缩放输出再加到输入。

### 类 `SwiGLUFFN`（L1896–L1916）

- `w12`: Linear `gc → e*ec`；chunk 两半，`silu(x1)*x2` → `w3` 到 `ec`（Transformer FFN 变体）。

### 类 `Residual`（L1919–L1937）

- 包装子模块 `m`（默认针对 `SwiGLUFFN` 的 `w3` 零初始化），`forward`: `x + m(x)`。

### 类 `SAVPE`（L1940–L2005）

#### `__init__`（L1943–L1968）

- 多尺度 `ch`：`cv1` 深卷积+上采样分支，`cv2` 轻 1×1 分支；`cv3/cv4/cv5/cv6` 融合到 `embed` 与 16 维空间提示维度。

#### `forward`（L1970–L2005）

- 多尺度特征对齐 concat；视觉提示掩码 `vp (B,Q,H,W)` 与空间特征做加权 softmax 聚合，得到 **L2 归一化** `(B,Q,embed)` 嵌入。

### 类 `Proto26`（L2008–L2052）

- 继承 `Proto`，构造函数以 `ch` 多尺度与 `nc` 语义类数。
- `feat_refine` 将 P4/P5 等到 P3 分辨率相加，`feat_fuse` 后进父类 `Proto` 得掩码原型；训练时可额外 `semseg` 分支输出语义 logits。
- `fuse()` 删除 `semseg` 便于推理。

### 类 `RealNVP`（L2055–L2125）

#### `__init__`（L2073–L2085）

- 6 层 coupling，`mask` 交替 `[0,1]/[1,0]`；每层 scale `s` 与 translate `t` 为 MLP；保留 `loc/cov` buffer 兼容旧权重。

#### `backward_p`（L2093–L2110）

- 2D 样本 `x (N,2)` 逆变换到潜变量 `z`，累加 log|det J|（耦合层闭式）。

#### `log_prob`（L2112–L2125）

- `backward_p` 得 `z` 与 `log_det`，加标准正态对数密度（2 维），用于姿态等 flow 损失。