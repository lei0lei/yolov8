# `ultralytics/solutions/config.py`

本模块用 `@dataclass` 定义 Vision AI Solutions 的统一配置容器 `SolutionConfig`。字段覆盖视频源、YOLO 模型路径、可视化开关、计数区域、姿态健身角度与关键点、分析图类型、模糊/裁剪/停车 JSON、速度估计标尺、推理与跟踪参数、OCR 识别器选型等。`update` 方法接受关键字参数：废弃键 `half` 经 `deprecation_warn` 映射为 `quantize=16`；未知键抛出 `ValueError` 并指向官方文档。

## 逐段说明

### 类 `SolutionConfig` 数据字段（第 11～107 行）

- 第 72～107 行：声明全部配置默认值（如 `tracker="tracktrack.yaml"`、`conf=0.25`、`colormap=cv2.COLORMAP_DEEPGREEN` 等）；`kpts` 用 `field(default_factory=...)` 默认 `[6,8,10]`。

### 方法 `SolutionConfig.update`（第 109～133 行）

- 第 121～125 行：若 kwargs 含 `half`，警告并设置 `quantize` 为 16 或依原值。
- 第 126～131 行：遍历 kwargs，存在同名 dataclass 字段则 `setattr`，否则抛错并附 solutions 参数文档 URL。
- 第 133 行：返回 `self` 以支持链式调用。