# `ultralytics/nn/backends/coreml.py`

本文件在 Apple 平台用 `coremltools` 加载 `.mlpackage` / `.mlmodel`，默认 Neural Engine（`CPU_AND_NE`），RT-DETR 头用 `ComputeUnit.ALL` 保精度。`forward` 处理动态 multi-array 与静态 PIL 图像两种导出约定，并将内置 NMS 的 `confidence`/`coordinates` 字典输出转为 `(B, n, 6)` 浮点数组。分割模型可能交换 proto 与 det 输出顺序。与统一接口对齐处：`AutoBackend` 在调用前常将 BCHW 转为 BHWC uint8/float。

## 逐段说明

### `CoreMLBackend` 类（第 17–88 行）

#### `load_model`（第 24–50 行）

`read_metadata` 决定 `compute_units`；加载失败回退 `CPU_ONLY`。读 spec 得 `input_name`、输入是否 `multiArrayType`（`dynamic`）；`apply_metadata`；成功加载且非仅 CPU 时 `prefetch=True`。

#### `forward`（第 52–88 行）

**输入**：`im` 已为 BHWC numpy（由上层转换）。

**动态 multi-array（第 64–67 行）**：转 NCHW，按 `batch` 分块 `predict`，沿 batch 维 concat 各输出键。

**静态图像列表（第 68–85 行）**：每图 `PIL.Image` uint8；若含 `confidence`（嵌入 NMS）：`xywh2xyxy` 乘 `[w,h,w,h]`，拼 `conf.max` 与 `argmax` 类别，pad 到 batch 内最大检测数，`(B,n,6)` fp32 返回。

**常规划分（第 80–85 行）**：合并 ndarray 输出；分类器仅 label+概率 dict 时堆成 `(B, num_classes)`。

**分割顺序（第 86–87 行）**：两输出且第二非 4D 时 `reversed`（proto 与 det 对调）。

## 与统一接口的对齐

| 阶段 | 行为 |
|------|------|
| 加载 | protobuf userDefined 或侧车元数据 → `names`、`task`。 |
| 前向 | 输入格式因导出而异（PIL vs multi-array）；输出统一为 `np.ndarray` 或 list。 |
| 输出 | 标准检测为 logits 张量；NMS-in-model 直接 6 列框+分+类；分割 `(det, proto)` 经顺序修正后与 PyTorch 一致。 |