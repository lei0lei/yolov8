# `ultralytics/models/yolo/classify/predict.py`

`ClassificationPredictor` 继承 `engine.predictor.BasePredictor`，将通用图像源推理改为分类流水线：ImageNet 式 resize/crop/normalize、无 NMS，输出 `Results.probs`。

相对基类：强制 `task=classify`；重写 `setup_source`、`pre_transform`、`preprocess`、`postprocess`，把检测的 letterbox+NMS 路径换成 torchvision 分类变换或拆分的主机/设备变换。

## 逐段说明

### 类 `ClassificationPredictor`（16–39）

文档说明支持 YOLO-cls 与 torchvision 权重。

### `__init__`（41–53）

调用 `super().__init__`；将 `self.args.task` 设为 `"classify"`，确保后续配置与回调按分类任务路由。

### `setup_source`（55–75）

- 懒导入 `torchvision.transforms`。
- `super().setup_source(source)` 建立数据集与批信息。
- 从 `self.model.model.transforms` 取已有变换；缺省（YAML 或未附带 transforms 的 checkpoint）则 `size=None`。
- 若 `size` 为 int 则扩为 `(size,size)`。
- 当已有 transforms 的尺寸与 `imgsz` 一致且后端为 `pt` 时复用；否则 `classify_transforms(self.imgsz)`。
- 若 compose 恰为 `Resize+CenterCrop+ToTensor+Normalize` 且通道为 3，则拆成 `host_transforms`（前两步）与 `device_transform`（后两步），便于 GPU 上只做归一化。

### `pre_transform`（77–79）

对每个 BGR `uint8` 图转 PIL，经 `host_transforms` 得 `uint8` numpy（仍留设备侧转换）。

### `preprocess`（81–89）

- 无 `device_transform` 且输入非张量：逐张 RGB PIL 经完整 `transforms`，`stack` 后迁至模型设备，按 `fp16` 转 half 或 float。
- 否则记录是否原本为张量，走 `super().preprocess`（letterbox 等），若非张量输入再套 `device_transform`。

### `postprocess`（91–110）

- 若 `orig_imgs` 非 list，用 `convert_torch2numpy_batch` 并 BGR 通道反转。
- 从 `preds` 取首元素（部分后端返回 tuple）并 `clone()` 避免复用缓冲区。
- 对每张图构造 `Results(orig_img, path=..., names=..., probs=pred)`，无 boxes。