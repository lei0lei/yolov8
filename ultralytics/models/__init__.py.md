# `ultralytics/models/__init__.py`

本文件是 `ultralytics.models` 包的公共门面：把 FastSAM、LLM、NAS、RT-DETR 及 YOLO 系列高层模型类再导出，供 `from ultralytics import YOLO, NAS, ...` 使用。SAM 未在包加载时同步导入，而是通过 `__getattr__` 懒加载，避免普通检测工作流被动拉起含 torchvision 的可选 SAM 依赖。`__all__` 中列出字符串形式的符号名，便于简化导入与静态分析。该文件本身不含训练或推理逻辑，只负责符号聚合与延迟加载 SAM。

## 逐段说明

### 行 3–7：子模块导入

从 `.fastsam`、`.llm`、`.nas`、`.rtdetr`、`.yolo` 分别导入 `FastSAM`、`LLM`、`NAS`、`RTDETR` 以及 `YOLO`、`YOLOE`、`YOLOWorld`。SAM 类不在此静态导入。

### 行 9：`__all__`

定义公开符号元组，包含 `"LLM"`、`"NAS"`、`"RTDETR"`、`"SAM"`、`"YOLO"`、`"YOLOE"`、`"FastSAM"`、`"YOLOWorld"`（与源码一致）。调用方可通过 `from ultralytics.models import *` 或属性访问获得这些名称。

### 行 12–19：`__getattr__(name)`（行 12）

**职责**：对未在模块命名空间内直接绑定的属性名做按需解析。

- **行 14–18**：若 `name == "SAM"`，在函数体内执行 `from .sam import SAM` 并返回 `SAM`，从而在首次访问 `ultralytics.models.SAM` 时才加载 SAM 子树。
- **行 19**：其他未知属性抛出 `AttributeError`，指明当前模块名与缺失属性名。