# `ultralytics/utils/callbacks/__init__.py`

本模块是 Ultralytics 训练集成回调包的公开入口，从 `base` 重新导出 `add_integration_callbacks`、`default_callbacks` 和 `get_default_callbacks`，供 `Trainer`/`Validator`/`Predictor`/`Exporter` 在构造时挂载默认钩子并按 `SETTINGS` 注入第三方日志回调。

## 逐段说明

### 模块级导出（第 3–5 行）

- 从相对导入 `.base` 拉取三个符号。
- `__all__` 仅声明这三个名称，控制 `from ultralytics.utils.callbacks import *` 的对外表面。