# `ultralytics/utils/export/paddle.py`

通过 **X2Paddle** 将 traced PyTorch 模型转为 **PaddlePaddle** 模型目录（`pytorch2paddle`，`jit_type="trace"`），Jetson 不支持。

## 逐段说明

### `torch2paddle`（第 12–54 行）

1. **assert** 非 Jetson；`check_paddle`、`check_requirements(x2paddle, six)`。
2. **Python 3.13 兼容**：临时将 `prim2code.prim_assert` 替换为空函数（池化 assert 在 PEP 667 下失败），**finally** 恢复原函数。
3. `pytorch2paddle(module, save_dir, input_examples=[im])`。
4. **若** 有 metadata：`YAML.save` 至 `metadata.yaml`。
5. 返回输出目录字符串。