# `ultralytics/models/sam/__init__.py`

本文件是 `ultralytics.models.sam` 包的公开入口，负责把 SAM 系列推理与模型类统一暴露给上层 `from ultralytics.models.sam import …` 调用方。它不做任何模型构建或前向计算，仅通过延迟的相对导入把 `SAM` 接口类与各代 Predictor 装配进包的命名空间。`__all__` 元组显式列出可被 `from module import *` 导出的符号，与 Ultralytics 其它子包保持一致的导出契约。Predictor 族覆盖 SAM1 基础预测、SAM2 图像/视频/动态交互，以及 SAM3 图像、语义、视频与其语义视频变体，使调用方无需深入 `predict.py` 即可按名称选用对应推理器。

## 逐段说明

### 模块许可头（第 1 行）

Ultralytics AGPL-3.0 许可证声明行，无运行时逻辑。

### 包级导入（第 3–13 行）

- **第 3 行**：从同级 `model` 子模块导入 `SAM` 类，作为用户面向的 Segment Anything 封装入口。
- **第 4–13 行**：从同级 `predict` 子模块批量导入八个 Predictor 类型：
  - `Predictor`：SAM1（ViT / Mobile-SAM）默认图像分割预测器；
  - `SAM2Predictor`、`SAM2VideoPredictor`、`SAM2DynamicInteractivePredictor`：SAM2 图像、视频与动态交互场景；
  - `SAM3Predictor`、`SAM3SemanticPredictor`、`SAM3VideoPredictor`、`SAM3VideoSemanticPredictor`：SAM3 交互、语义提示、视频跟踪及视频语义组合。
- 这些导入在包首次被 import 时执行，会连带加载 `predict.py` 及其依赖链。

### 公开符号列表 `__all__`（第 15–25 行）

- **第 15–24 行**：元组包含与上文导入完全一致的九个名称（一个 `SAM` + 八个 Predictor），顺序为 SAM 优先、其余按 SAM2 再 SAM3 分组。
- **第 25 行**：行尾注释说明该结构用于限定 `import *` 的可见符号集合。
- 未列入 `__all__` 的 `build.py`、`build_sam3.py`、`amg.py` 等仍可通过子模块路径访问，但不属于本包默认公开面。