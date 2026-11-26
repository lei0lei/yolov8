# `ultralytics/solutions/__init__.py`

本模块聚合 Ultralytics Solutions 各场景类的公开入口。从子模块导入健身监测、分析图表、测距、热力图、实例分割、模糊/计数/裁剪、OCR、停车、队列、区域计数、安防告警、相似度搜索、测速、Streamlit 推理、区域跟踪、 VisionEye 等实现类。`__all__` 与导入列表一致，供 `ultralytics.cfg.handle_yolo_solutions` 通过 `getattr(solutions, SOLUTION_MAP[name])` 动态实例化，也支持用户 `from ultralytics.solutions import ObjectCounter` 直接引用。

## 逐段说明

### 导入块（第 3～20 行）

逐行 `from .<module> import <Class>`，无附加逻辑。

### `__all__`（第 22～43 行）

元组列出全部对外类名，顺序与文档 solutions 列表对应。