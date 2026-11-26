# `ultralytics/utils/export/coreml.py`

**CoreML**：`torch2coreml` 将 traced 模型转为 **`.mlpackage`/`.mlmodel`**；`IOSDetectModel` 为 iOS 检测输出归一化与类维填充；`pipeline_coreml` 拼接 Apple NMS 管道。

## 逐段说明

### `IOSDetectModel`（第 18–56 行）

- **init**：保存内层 model、类数；方形输入用标量 `1/w` 归一化，否则 4 向量 `[1/w,1/h,1/w,1/h]`。
- **forward**：取 `model(x)[0]` 转置 split 为 xywh 与 cls；MLProgram 且 nc 非 80 倍数时 pad cls 至 80 倍数；返回 cls 与 `xywh*normalize`。

### `pipeline_coreml`（第 59–174 行）

1. 取 spec；**mlmodel** 时手设 multiArray 输出 shape。
2. 读输入尺寸与类数；类名不足 nc 时临时补数字名（NMS bug workaround）。
3. 克隆输出为 NMS 子模型输入/输出；重命名为 confidence/coordinates。
4. 配置 `nonMaximumSuppression`：iou/conf 阈值、perClass topK、类标签字符串。
5. `Pipeline` 串联原模型 + NMS；修正 datatype 链；复制 metadata 与用户阈值说明。
6. 构造 `MLModel`（`weights_dir` 供 MLProgram）；设置 input/output 描述。

### `_coreml_gather`（第 177–179 行）

- batch 维 arange 与 index 组合高级索引，避免 MIL int64 gather 问题。

### `torch2coreml`（第 182–276 行）

**准备（212–217）**

- Detect 模块替换 `_gather`；trace `model.eval()`；`bits` 由 quantize 解析 8/16/32。

**convert_kwargs（224–251）**

- neuralnetwork vs mlprogram；classifier config 可选。
- MLProgram + FP32：`compute_precision=FLOAT32`。
- RT-DETR：`FP16ComputePrecision` 排除 linear/gather 类算子。
- NMSModel 非 obb：按 torch scope 将 div/mul/add/cat/torchvision::nms 保持 FP32。

**量化（253–265）**

- mlmodel：`quantize_weights` linear。
- mlprogram int8：kmeans palettizer；跳过大且含非有限 的 const。

**元数据与保存（267–276）**

- 弹出 description/author/license/version 到 CoreML 字段；其余进 user_defined_metadata。
- **若** `output_file`：`save`；返回 `ct_model`。