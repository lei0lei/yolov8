# `ultralytics/optim/__init__.py`

本包聚合 Ultralytics 训练用自定义优化器实现。当前仅导出 MuSGD（Muon 正交更新与 SGD 动量混合）优化器类。模块顶部 docstring 说明用途为「Optimizers for Ultralytics model training」。通过相对导入从 `muon` 子模块引入 `MuSGD`，并以 `__all__ = ["MuSGD"]` 限定 `from ultralytics.optim import *` 的公开符号，与训练配置里选用 MuSGD 的入口保持一致。

## 逐段说明

### 包文档与导入（第 1～6 行）

- 第 1 行：Ultralytics 许可证头。
- 第 2 行：模块级说明字符串。
- 第 4 行：`from .muon import MuSGD` 绑定混合优化器实现。
- 第 6 行：`__all__` 仅包含 `MuSGD`，对外 API 单一明确。