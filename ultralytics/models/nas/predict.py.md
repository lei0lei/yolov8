# `ultralytics/models/nas/predict.py`

定义 `NASPredictor`，继承 YOLO 检测 `DetectionPredictor`，不在此文件重写 `postprocess` 或 `preprocess`。YOLO-NAS（SuperGradients）原始前向输出格式在 `NAS._load` 中已通过绑定自定义 `forward` 转为 Ultralytics 检测张量布局；Predictor 因此复用标准检测 NMS、尺度还原与 `Results` 封装。用户通常通过 `NAS(...).predict(...)` 间接使用本类，而非手动构造。

## 逐段说明

### 行 3–4：导入与类声明

从 `ultralytics.models.yolo.detect.predict` 导入 `DetectionPredictor`。`NASPredictor(DetectionPredictor)` 为空子类，无新增方法或属性，文档字符串说明其负责 NAS 权重推理后的后处理职责及典型用法（经 `model.predictor` 访问）。