# `ultralytics/models/nas/val.py`

定义 `NASValidator`，继承 `DetectionValidator`，同样不在此扩展验证循环。NAS 权重经 `NAS._load` 标准化后，`names`、`stride` 与检测输出格式与 Ultralytics 检测验证器假设一致，故直接复用 mAP 计算、NMS 与 COCO 指标路径。`__all__` 仅导出 `NASValidator` 名称供类型提示或显式导入。

## 逐段说明

### 行 3：导入

从 `ultralytics.models.yolo.detect` 导入 `DetectionValidator`。

### 行 5：`__all__`

列表形式声明 `["NASValidator"]`。

### 行 8–26：`NASValidator` 类

继承 `DetectionValidator`，无自定义 `__init__` 或钩子；类文档说明在 `NAS.val()` 内部实例化、对 NAS 原始预测做 NMS 后产出最终框集。