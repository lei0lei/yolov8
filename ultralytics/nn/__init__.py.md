# `ultralytics/nn/__init__.py`

本文件是 `ultralytics.nn` 包的公共入口，不定义任何类或函数，仅通过相对导入从 `tasks` 子模块再导出模型构建与加载相关的符号。对外接口即 `__all__` 中列出的十个名称：`BaseModel`、`ClassificationModel`、`DetectionModel`、`SegmentationModel`，以及 `guess_model_scale`、`guess_model_task`、`load_checkpoint`、`parse_model`、`torch_safe_load`、`yaml_model_load`。上层代码通常通过 `from ultralytics.nn.tasks import …` 直接引用，或通过 `ultralytics.nn` 包名访问（若外部重导出）。`engine/model.py` 与 `models/yolo/model.py` 在实例化 YOLO 时依赖各任务模型类；训练器、导出器与 `nn/backends/pytorch.py` 调用 `load_checkpoint`；测试与 CLI 间接使用 `guess_model_task` 与 `yaml_model_load`。该文件刻意保持最小化，避免在 import `ultralytics.nn` 时加载 `autobackend`、`text_model` 等较重子模块。

## 逐段说明

### 第 1 行：许可证头（行 1）

Ultralytics 标准 AGPL-3.0 许可证注释，对运行时无行为影响。

### 第 3–14 行：相对导入（行 3–14）

从当前包的 `tasks` 模块导入九项符号：四个模型基类/任务类，六个工具函数。导入在包首次被加载时执行，会触发 `tasks.py` 的完整加载（含大量 `nn.modules` 与损失类依赖），因此 `import ultralytics.nn` 的开销主要来自 `tasks`，而非本文件本身。

### 第 16–27 行：`__all__` 元组（行 16–27）

显式声明包的公共 API，供 `from ultralytics.nn import *` 及文档工具识别。元组内容与导入列表一一对应，未包含 `OBBModel`、`PoseModel` 等其它任务类——那些需从 `ultralytics.nn.tasks` 直接导入。