# `ultralytics/utils/export/deepx.py`

使用 **DEEPX DX-Compiler**（`dx_com`）将 **ONNX** 量化为 **DEEPX** 部署目录，含校准 JSON 与 BGR→RGB 预处理链。

## 逐段说明

### `onnx2deepx`（第 12–78 行）

1. `dx_com` 无 GUI opencv 依赖安装；解析其 requirements 排除 `opencv-python`。
2. 输出 `{stem}_deepx_model`；写 `config.json`：输入 shape、EMA 校准 100 步、数据集路径与 pad/255/BGR2RGB/transpose 管线。
3. `dx_com.compile`：`opt_level=int(optimize)`。
4. 可选 metadata YAML；返回路径。