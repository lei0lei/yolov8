# `ultralytics/utils/export/imx.py`

**Sony IMX500**：MCT 混合精度 PTQ/GPTQ → 量化 PyTorch → **NMSWrapper** → ONNX（`pytorch_export_model`）→ **`imxconv-pt`** 二进制与 `labels.txt`。仅 YOLOv8n/YOLO11n 固定层数配置。

## 逐段说明

### `MCT_CONFIG`（第 23–60 行）

- 按 `YOLO11`（含 C2PSA）或 `YOLOv8`、task 定义手动 FP16 激活层名、权重内存上限、期望 `n_layers` 模块总数。

### `FXModel`（第 63–119 行）

- `copy_attr` 复制原模型；显式设 `self.model` 为 backbone 列表。
- **forward**：按 YAML 拓扑 `m.f` 取特征；遇 Detect 绑 `_inference` 与 `make_anchors`；Pose/Segment 绑专用 forward；串联输出。

### `_inference`（第 122–125 行）

- DFL+decode×strides；返回 boxes 与 sigmoid scores 转置格式。

### `pose_forward`（第 128–142 行）

- 拼 cv4 关键点；Pose26 五维裁到三维；`Detect.forward` + `kpts_decode` 转置。

### `segment_forward`（第 145–151 行）

- proto + mask coeff + Detect 输出四元组。

### `NMSWrapper`（第 154–211 行）

- **forward**：`multiclass_nms_with_indices`；pose/segment 用 indices gather 关键点或 mask coeff；detect 额外返回 `n_valid`。

### `torch2imx`（第 214–382 行）

**环境（262–282）**

- Java≥17（按 OS 装 JRE）；MCT、edge-mdt、imx500-converter 等依赖。

**量化（285–337）**

- 解析 task 的 `MCT_CONFIG`；层数不在 `n_layers` 集 → ValueError。
- 对手动层名设 activation 16bit；`CoreConfig` + `ResourceUtilization`。
- **gptq 分支**：`pytorch_gradient_post_training_quantization`；**else** `pytorch_post_training_quantization`。

**NMS 与 ONNX（339–363）**

- 非 classify：`NMSWrapper` 包装。
- `onnx_export_patch` 下 `pytorch_export_model`；ONNX metadata_props。

**imxconv（365–381）**

- 优先 venv 内 `imxconv-pt`；`subprocess` 编译；写 `labels.txt` 类名行。
- 返回 output_dir。