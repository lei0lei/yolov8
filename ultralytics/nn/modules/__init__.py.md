# `ultralytics/nn/modules/__init__.py`

本文件是 Ultralytics 神经网络**构建块的统一入口**，不负责实现具体算子，而是把 `block`、`conv`、`head`、`transformer` 子包里的公开类聚合到一起。YAML 解析模型时，`parse_model` 通过 `from ultralytics.nn.modules import Conv, C2f, Detect` 这类符号把字符串层名绑定到实际 `nn.Module` 子类。检测与分割的**任务头**（`Detect`、`Segment`、`OBB` 等）与骨干用的**卷积/CSP/注意力块**在同一命名空间导出，保证一处维护、`__all__` 与 IDE 补全一致。用户或工具（如 Netron 可视化）也可直接从本包按需导入单个层做单元导出。文档字符串中的示例展示了如何把 `Conv` 导出为 ONNX。整体上，该模块在 YOLO 系列网络里扮演**层注册表与公共 API 门面**的角色，连接配置描述与底层 `modules` 实现。

## 逐段说明

### 模块文档与示例（约 L2–L17）

说明本包提供卷积块、注意力、Transformer 部件及检测/分割头等组件；附带用 `Conv` 做 ONNX 导出的 doctest 风格示例。

### 从 `block` 导入（L19–L61）

一次性导入 CSP 系列（`C1`–`C3k2`、`C2f`、`C2fPSA` 等）、池化（`SPP`、`SPPF`、`SPPELAN`）、下采样（`AConv`、`ADown`、`SCDown`）、Rep 结构、HGNet 块（`HGStem`、`HGBlock`）、注意力与视觉语言相关块（`Attention`、`ImagePoolingAttn`、`ContrastiveHead`）、辅助层（`DFL`、`Proto`、`CBLinear`、`CBFuse`）、`TorchVision` 骨干封装等。这些类多出现在 backbone/neck 的 YAML 行里，负责特征提取、融合与部分任务专用预处理。

### 从 `conv` 导入（L62–L77）

导入最基础的**卷积族**与路由算子：`Conv`/`DWConv`/`GhostConv`/`RepConv`、`Focus`、转置卷积、`CBAM` 注意力，以及 FPN 常用的 `Concat` 和从多输出中选一路的 `Index`。几乎所有特征图尺寸与通道变换都依赖这一组。

### 从 `head` 导入（L78–L96）

导入各任务**检测与解码头**：`Detect`（及 `v10Detect`、`WorldDetect`、`YOLOEDetect`）、`Segment`/`Segment26`/`YOLOESegment`、`Pose`/`Pose26`、`OBB`/`OBB26`、`Classify`、`Depth`、`SemanticSegment`、`RTDETRDecoder`、`LRPCHead` 等。它们接在 neck 多尺度特征之后，输出框、掩码、关键点或类别 logits。

### 从 `transformer` 导入（L97–L108）

导入 RT-DETR、AIFI 等所需的 Transformer 构件：`TransformerEncoderLayer`、`DeformableTransformerDecoder`、`MSDeformAttn`、`LayerNorm2d`、`MLP` 等，供 `C3TR` 或解码器分支使用。

### `__all__` 元组（L110–L193）

显式列出对外公开符号名，与上述 import 对齐（未包含部分仅内部使用的名字）。训练、导出和第三方扩展应以此列表为准判断稳定 API。