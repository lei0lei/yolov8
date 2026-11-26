# `ultralytics/data/__init__.py`

本文件是 `ultralytics.data` 包的公开入口，职责是把数据子系统里最常用的 Dataset 类与工厂函数统一再导出，供训练、验证与推理代码 `from ultralytics.data import …` 使用。它不实现任何数据逻辑，仅组织 `__all__` 与相对导入。对外接口包括 `BaseDataset`、多种 YOLO 系数据集类，以及 `build_dataloader`、`build_yolo_dataset`、`build_grounding`、`load_inference_source`。训练侧一般由 `ultralytics.engine.trainer` 经 `build_yolo_dataset` / `build_dataloader` 间接引用；预测侧由 `engine.predictor` 经 `load_inference_source` 解析输入源。第三方集成或脚本也可直接导入这些符号以保持与官方 CLI 一致。

## 逐段说明

### 许可证头（第 1 行）

Ultralytics AGPL-3.0 标准文件头，无执行逻辑。

### 导入分组（第 3–13 行）

- **基类**：`from .base import BaseDataset` — 所有 YOLO 文件型数据集的父类。
- **构建与推理入口**：`from .build import build_dataloader, build_grounding, build_yolo_dataset, load_inference_source` — DataLoader 工厂、任务数据集工厂、Grounding 专用工厂、预测源路由。
- **具体数据集**：`from .dataset import …` — 检测/分割/语义/多模态/拼接等 `Dataset` 实现类。

### `__all__`（第 15–28 行）

显式列出包级公开符号，与上述导入一一对应，便于 `from ultralytics.data import *` 时边界清晰，也方便静态工具识别 API 面。