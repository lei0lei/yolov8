# `ultralytics/models/fastsam/__init__.py`

FastSAM 子包入口，从同目录导出模型门面类 `FastSAM` 以及引擎侧使用的 `FastSAMPredictor`、`FastSAMValidator`。本文件不实现分割逻辑，仅把三类符号暴露给 `from ultralytics.models.fastsam import ...` 与 `from ultralytics import FastSAM` 的导入链。与 YOLO 各任务子包类似，Predictor/Validator 在 `task_map` 中由 `FastSAM` 类按任务字符串懒绑定。

## 逐段说明

### 行 3–5：相对导入

从 `.model` 导入 `FastSAM`；从 `.predict` 导入 `FastSAMPredictor`；从 `.val` 导入 `FastSAMValidator`。

### 行 7：`__all__`

以元组形式公开上述三个符号名，供 `__all__` 约束下的 `import *` 与文档工具识别。