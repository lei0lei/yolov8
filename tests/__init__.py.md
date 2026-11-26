# `tests/__init__.py`

本模块不是测试用例，而是整个 `tests/` 包共享的常量与配置中心。它为 pytest 用例提供统一的模型权重路径、输入图像源、任务—模型—数据集三元组，以及 Solutions 测试所需资源的文件名映射。路径中刻意包含空格（`path with spaces`），用于验证 Ultralytics 在含空格路径下的加载与推理行为。模块在导入时调用 `checks.cuda_is_available()` 与 `checks.cuda_device_count()`，供 GPU 相关测试做条件跳过或参数化。`TASK_MODEL_DATA` 按任务字母序排列，与 `TASKS` 注册表一致；`MODELS` 还额外纳入灰度专用权重。

## 逐段说明

### 模块级常量（第 7–28 行）

- **MODEL**：指向 `WEIGHTS_DIR` 下带空格子目录中的 `yolo26n.pt`，是多数快速 smoke 测试的默认权重。
- **CFG / SOURCE / SOURCES_LIST**：默认 YAML 配置、单张 `bus.jpg`，以及文件、目录、通配符三种 predict 源形态列表。
- **CUDA_IS_AVAILABLE / CUDA_DEVICE_COUNT**：运行环境 CUDA 探测结果，供 `test_cuda.py` 等使用。
- **TASK_MODEL_DATA**：`(task, weights_path, data_yaml)` 有序列表，覆盖全部官方任务。
- **MODELS**：各任务默认 checkpoint 文件名列表，外加 `yolo11n-grayscale.pt`。
- **SOLUTION_ASSETS**：键名为逻辑别名、值为远端资产文件名的字典，由 `conftest.solution_assets` fixture 按需下载。

### `__all__`（第 30–38 行）

显式导出 `CFG`、`CUDA_DEVICE_COUNT`、`CUDA_IS_AVAILABLE`、`MODEL`、`SOLUTION_ASSETS`、`SOURCE`、`SOURCES_LIST`，约束 `from tests import *` 的公开表面。