# `ultralytics/utils/export/litert.py`

**LiteRT（Google）**：`torch2litert` 经 **litert_torch** 转 **`.tflite`**，支持静态 INT8、W8A16、动态 W8A32，可选 **i.MX95 Neutron** 二次编译（`tflite2nxp`）。

## 逐段说明

### `_litert_grouped_topk`（第 19–22 行）

- 委托 `Detect._grouped_topk`，indices 转 int32（GPU delegate 要求）。

### `_litert_gather`（第 25–29 行）

- 用展平 + `index_select` 模拟 gather，避免 GATHER_ND。

### `torch2litert`（第 32–151 行）

**量化模式（71–75）**

- static_int8 / static_int16 无 calibration → ValueError。

**模型包装（80–96）**

- 检测/分割/姿态/obb 且非 end2end、非 RT-DETR：套 `_NormalizeCoords`。
- Detect：替换 `_grouped_topk`、`_litert_gather`。
- 注册 `index_select` → `tfl.gather` 预分解。

**导出（99–104）**

- `litert_torch.convert` + `export` 到带后缀的 `.tflite`。

**ai-edge-quantizer（106–145）**

- static：`load_quantization_recipe` static_wi8_ai8 或 wi8_ai16；收集校准 numpy；I/O 设 `NO_QUANTIZE`。
- static INT8 + Classify/Detect：regex 排除 head 与 NormalizeCoords 输出；Classify linear 动态 8bit 权重配置。
- `calibrate` + `quantize` 覆盖写 tflite。
- dynamic：`recipe.dynamic_wi8_afp32()` 直接 quantize。

**元数据（147–151）**

- 以 zip append 方式写入 `metadata.json` 进 tflite 文件。

### `tflite2nxp`（第 154–185 行）

- 依赖 `eiq-neutron-sdk==3.3.0`；子进程调用 neutron_compiler，`--target imx95`。
- 同样 append metadata.json 到输出。