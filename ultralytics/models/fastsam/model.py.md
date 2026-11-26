# `ultralytics/models/fastsam/model.py`

`FastSAM` 继承 engine `Model`，将任务固定为 `segment`，仅支持 `.pt` 预训练权重（拒绝 `.yaml`）。在 `predict` 层把框/点/标签/文本提示打包为 `prompts` 字典传给引擎 Predictor。`task_map` 将 segment 任务映射到 `FastSAMPredictor` 与 `FastSAMValidator`。

## 逐段说明

### 行 8–11：导入

从 `ultralytics.engine.model` 导入 `Model`；从本包 `.predict`、`.val` 导入 Predictor 与 Validator 类。

### `FastSAM`（行 14）

文档描述属性 `model`、`task` 及 `predict`、`task_map` 职责。

### `__init__(self, model="FastSAM-x.pt")`（行 38）

- **行 44–45**：若路径字符串为 `"FastSAM.pt"`，替换为 `"FastSAM-x.pt"` 默认权重别名。
- **行 46**：断言后缀不是 `.yaml`/`.yml`，否则 FastSAM 不支持 YAML 结构训练。
- **行 47**：`super().__init__(model=model, task="segment")` 走通用 Model 加载与任务绑定。

### `predict(self, source=None, stream=False, bboxes=None, points=None, labels=None, texts=None, **kwargs)`（行 49）

- **行 77**：组装 `prompts` 字典，键为 `bboxes`、`points`、`labels`、`texts`。
- **行 78**：调用 `super().predict(source, stream, prompts=prompts, **kwargs)`，由引擎创建/复用 `FastSAMPredictor` 并传入 prompts。

### `task_map` 属性（行 80）

- **行 83**：返回 `{"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}`。