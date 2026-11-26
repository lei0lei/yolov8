# `ultralytics/models/yolo/__init__.py`

该文件是 YOLO 任务子包的公共入口，不重写 engine 中的训练/验证/预测逻辑，只做子模块聚合与门面类再导出。相对 engine 基类，本文件本身不包含 Trainer/Predictor/Validator 实现，而是把各任务子包与三个高层 Model 类暴露给 `from ultralytics.models.yolo import ...` 调用方。导入时按任务拆开为独立子包，便于 `YOLO.task_map` 按字符串键懒加载对应实现。

## 逐段说明

### 行 3：子包导入

从 `ultralytics.models.yolo` 导入九个任务命名空间：`classify`、`depth`、`detect`、`obb`、`pose`、`segment`、`semantic`、`world`、`yoloe`。这些子包各自持有该任务的 Predictor/Trainer/Validator，供 `model.py` 中 `task_map` 属性引用。

### 行 5：门面类导入

从同目录 `model.py` 导入 `YOLO`、`YOLOE`、`YOLOWorld` 三个继承自 `engine.model.Model` 的统一入口类。

### 行 7–20：`__all__`

公开符号为上述三个类加上九个任务子包名，与 `from ultralytics.models.yolo import YOLO, detect, ...` 的推荐用法一致。