# `ultralytics/utils/export/ncnn.py`

使用 **PNNX** 将 PyTorch 模型导出为 **NCNN**（`.param`/`.bin`），固定 PNNX 版本，导出后校验 NCNN 可加载。

## 逐段说明

### `torch2ncnn`（第 12–76 行）

1. `check_requirements`：`ncnn --no-deps`、`pnnx==20260526`。
2. 配置 `ncnn_args` 与 `pnnx_args` 输出路径；`mkdir`。
3. `pnnx.export(..., fp16=(quantize==16), device=device_type)`。
4. 删除 debug 与中间 pnnx 文件。
5. `ncnn.Net().load_param/load_model`：**任一步非 0** 则 `RuntimeError`。
6. 可选 `metadata.yaml`；返回目录路径。