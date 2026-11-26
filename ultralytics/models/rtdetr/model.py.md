# `ultralytics/models/rtdetr/model.py`

`RTDETR` 为百度 RT-DETR 的门面类，继承 `Model`，任务为 `detect`。构造时要求 `torch>=1.11`。`task_map` 在 detect 下绑定 Predictor、Validator、Trainer 及 `RTDETRDetectionModel` 网络类，支持 `.pt`/`.yaml` 与训练闭环。

## 逐段说明

### 行 15–21：导入

`Model`、`RTDETRDetectionModel`、`TORCH_1_11` 版本门；本包 predict/train/val。

### `RTDETR.__init__(self, model="rtdetr-l.pt")`（行 43）

- **行 50**：`assert TORCH_1_11`，不满足则抛错提示需 torch>=1.11。
- **行 51**：`super().__init__(model=model, task="detect")`。

### `task_map` 属性（行 53）

返回字典：`detect` 键下含 `predictor: RTDETRPredictor`、`validator: RTDETRValidator`、`trainer: RTDETRTrainer`、`model: RTDETRDetectionModel`。