# `ultralytics/models/nas/__init__.py`

YOLO-NAS 子包入口，再导出 `NAS` 门面、`NASPredictor` 与 `NASValidator`。检测后处理与验证指标由 Predictor/Validator 在各自模块中继承 YOLO 检测引擎实现；本文件只做符号聚合，便于 `NAS.task_map` 与外部 `from ultralytics import NAS` 解析类路径。

## 逐段说明

### 行 3–5：相对导入

从 `.model` 导入 `NAS`；从 `.predict` 导入 `NASPredictor`；从 `.val` 导入 `NASValidator`。

### 行 7：`__all__`

公开 `"NAS"`、`"NASPredictor"`、`"NASValidator"` 三个名称。