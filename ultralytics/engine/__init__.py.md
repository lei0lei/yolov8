# `ultralytics/engine/__init__.py`

本文件是 Ultralytics 推理与训练引擎子包 `ultralytics.engine` 的包标识模块，除 AGPL 许可证头外不包含任何可执行代码、导出符号或 `__all__` 定义。实际对外能力分散在同目录下的 `model.py`、`predictor.py`、`validator.py`、`trainer.py`、`exporter.py` 等模块中；用户代码通常通过 `from ultralytics import YOLO` 间接使用引擎，而 `YOLO` 等子类在 `ultralytics/models` 中继承 `engine.model.Model` 并提供 `task_map`。CLI 的 `yolo` 入口在解析 `mode` 后同样落到这些引擎类。因此本文件职责仅限于标记目录为 Python 包，便于相对导入 `ultralytics.engine.model` 等形式。

## 逐段说明

### 第 1 行：许可证头

文件仅含一行 Ultralytics AGPL-3.0 许可证注释。无 `import`、无类、无函数；导入该包时不会执行额外逻辑，也不会向命名空间注册符号。