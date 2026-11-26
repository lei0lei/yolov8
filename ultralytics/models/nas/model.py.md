# `ultralytics/models/nas/model.py`

`NAS` 门面封装 YOLO-NAS（SuperGradients）权重：禁止 YAML，仅 `.pt`。重写 `_load` 以下载权重、包装 `forward` 为 Ultralytics 检测输出格式 `(B, 4+nc, N)`，并注入 `fuse`、`stride`、`names`、`task` 等 engine/export 期望的属性。`info` 委托 `model_info`；`task_map` 指向 NAS 专用 Predictor/Validator。

## 逐段说明

### 行 11–19：导入

`Model`、配置与下载工具、`torch_load`、`model_info`；本包 predict/val 类。

### `NAS.__init__(self, model="yolo_nas_s.pt")`（行 45）

- **行 51**：断言非 yaml/yml。
- **行 52**：`super().__init__(model, task="detect")`。

### `_load(self, weights, task=None)`（行 54）

- **行 61**：`torch_load(attempt_download_asset(check_model_file_from_stem(weights)), weights_only=SAFE_LOAD)` 得到 SG 模块赋给 `self.model`。

- **嵌套 `forward(m, x)`（行 63–67）**：
  - 调用 `type(m).forward(m, x)` 得 `y`；
  - 若非 JIT trace：`b, scores = y[0]`，否则整段 `y` 当作 `b, scores`；
  - 将角点+尺寸转为 xywh 中心格式并拼接类别分数，`permute(0,2,1)` 得到 Ultralytics 布局。

- **行 70–79**：`MethodType` 绑定 `forward`、`fuse`（恒等返回 `m`）、`stride=tensor([32])`、`names` 由 `_class_names` 枚举、`is_fused` 恒 False、`yaml` 空 dict、`pt_path`、`task="detect"`、`args` 合并默认 cfg 与 overrides、`eval()`。

- **行 80**：`self.model_name = weights`。

### `info(self, detailed=False, verbose=True, imgsz=640)`（行 82）

返回 `model_info(self.model, ...)` 四元组（层数、参数量等）。

### `task_map`（行 95）

返回 detect 任务的 predictor/validator 类映射。