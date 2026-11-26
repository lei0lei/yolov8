# `ultralytics/models/rtdetr/__init__.py`

RT-DETR 子包公共入口，导出高层 `RTDETR` 模型类以及 `RTDETRPredictor`、`RTDETRValidator`。训练器 `RTDETRTrainer` 未列入 `__all__`，但仍可通过 `RTDETR.task_map["detect"]["trainer"]` 由引擎按需实例化。本文件不含 Transformer 检测头或损失实现，仅完成门面与推理/验证类的再导出。

## 逐段说明

### 行 3–5：相对导入

从 `.model` 导入 `RTDETR`；从 `.predict` 导入 `RTDETRPredictor`；从 `.val` 导入 `RTDETRValidator`。

### 行 7：`__all__`

元组形式导出 `"RTDETR"`、`"RTDETRPredictor"`、`"RTDETRValidator"`。