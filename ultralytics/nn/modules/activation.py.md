# `ultralytics/nn/modules/activation.py`

本文件目前提供 Ultralytics 可选用的**可学习激活函数**模块，与固定 `SiLU`/`ReLU` 不同，可在训练中自适应调整非线性形状。AGLU（Adaptive Gated Linear Unit）来自第三方工作，通过可训练标量 `λ`、`κ` 与 Softplus 门控组合，在部分实验型 backbone 或替换默认激活的 YAML 实验中出现。激活不改变张量 rank，输入输出形状一致 `(…)`，便于插入任意卷积块之后。该文件体量小，职责单一：集中存放非标准 `nn.Module` 激活，避免污染 `conv.py`。

## 逐段说明

### 类 `AGLU`（L8–L54）

#### `__init__`（L34–L39）

- `self.act`：`nn.Softplus(beta=-1.0)`，对中间量做平滑正部。
- `self.lambd`、`self.kappa`：各为形状 `(1,)` 的 `nn.Parameter`，`uniform_` 初始化；可选 `device`/`dtype` 与主网络一致。
- 无子卷积；通道维不参与，纯逐元素激活。

#### `forward`（L41–L54）

输入 `x` 任意形状 `(...)`。

1. `lam = clamp(self.lambd, min=1e-4)`，防止除零。
2. 计算 `(1/lam) * act(kappa * x - log(lam))` 再 `exp`。
3. 输出与 `x` **同 shape**，数值为 AGLU 公式给出的门控指数变换。