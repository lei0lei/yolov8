# `ultralytics/utils/export/stedgeai.py`

**STM32 ST Edge AI** 导出前模型包装：在调用 TensorFlow 导出路径前，对 **Semantic/Depth** 与 **Detect/Pose** 头打补丁，使整图 INT8 量化时坐标与 logits 范围可共享 scale。

## 逐段说明

### `stedgeai_wrapper`（第 11–30 行）

- 遍历 `model.modules()`：
  - **SemanticSegment**：`forward` 仅返回 `classifier(x[0])`（低分辨率 logits）。
  - **Depth**：`forward` 返回仿射后的 log-depth `cal_a * log_depth(x) + cal_b`。
- 最后 `return tf_wrapper(model)`（Detect/Pose TF 友好解码，见 `tensorflow.py`）。
- 无新增函数；返回同一实例引用。