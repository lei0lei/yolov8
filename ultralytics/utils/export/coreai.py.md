# `ultralytics/utils/export/coreai.py`

用 **coreai-torch** 将 `torch.export` 程序转为 Apple **Core AI `.aimodel`** 资源，可选 FP16，分解 `aten.remainder.Scalar` 以兼容 lowering。

## 逐段说明

### `_remainder_scalar`（第 13–21 行）

- 整数 dtype：用 `div(..., rounding_mode="floor")` 实现取模语义。
- 浮点：`x - s * floor(x/s)`（与 fmod 在负值上行为不同）。

### `torch2coreai`（第 24–80 行）

1. `check_requirements("coreai-torch>=0.4.3")`。
2. **若** `quantize==16`：模型与示例输入 half。
3. `torch.export.export` → 自定义 decomp 表替换 remainder → `run_decompositions`。
4. `TorchConverter.add_exported_program`：入口 `main`，I/O 名 `images`/`output*`。
5. `AIModelAssetMetadata` 填 author/license/description 与自定义键；`program.save_asset`。
6. 返回输出路径。