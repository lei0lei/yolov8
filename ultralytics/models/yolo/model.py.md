# `ultralytics/models/yolo/model.py`

定义 `YOLO`、`YOLOWorld`、`YOLOE` 三个 `engine.model.Model` 子类：自动路由专用类型、`task_map` 绑定各任务 nn/tasks 与 yolo 子包 Trainer/Validator/Predictor；YOLOE/YOLOWorld 额外提供开放词汇与 prompt 嵌入 API。

相对 engine：不实现训练循环，扩展构造时分派、任务表、以及 YOLOE 的 val/predict 视觉提示分支。

## 逐段说明

### 类 `YOLO`（32–143）

#### `__init__`（60–95）

- `path` 来自 str/Path 模型路径；`yaml` 判断后缀 `.yaml/.yml`。
- YAML 且 stem 含 `-world` → `YOLOWorld(...)` 实例。
- YAML 且 `yoloe` in stem → `YOLOE(..., task)`。
- YAML 且 `rtdetr` → `RTDETR(path)`。
- 否则 `super().__init__` 标准 YOLO；读 head 名：最后一层 `_get_name()`，导出模型则从 `BaseBackend.read_metadata` 取 `head`。
- head 含 `RTDETR` → `RTDETR(self)`；`WorldModel`/`YOLOEModel` → 对应 `YOLOWorld`/`YOLOE(self)`。
- 若 `new_instance` 非空：`__class__` 与 `__dict__` 替换为派发实例（透明代理）。

#### `task_map` 属性（97–143）

七任务字典：classify/detect/segment/pose/obb/depth/semantic，每项含 model 类与四件套 yolo 子模块引用。

### 类 `YOLOWorld`（146–209）

#### `__init__`（171–186）

`super(..., task="detect")`；无 `model.names` 时从 `coco8.yaml` 加载默认 COCO 名。

#### `task_map`（188–198）

仅 detect：`WorldModel`、WorldValidator、DetectionPredictor、WorldTrainer。

#### `set_classes`（200–209）

`model.set_classes`；`model.names=list(classes)`；`predictor=None` 强制下次重建 predictor。

### 类 `YOLOE`（212–568）

#### `__init__`（249–259）

`super` 保留自动 task。

#### `task_map`（261–277）

detect 与 segment 两套 YOLOEModel/YOLOESegModel 与 yoloe 验证/训练器。

#### `get_text_pe`（279–289）

断言 `YOLOEModel`；委托 `model.get_text_pe(texts)`。

#### `get_visual_pe`（291–311）

同上委托视觉嵌入。

#### `set_vocab`（313–338）

断言类型；`check_class_names`；`predictor=None`；`model.set_vocab(vocab, names, one2one_vocab)` 重参数化为无 prompt 头。

#### `get_vocab`（340–351）

`predictor=None`；返回 `model.get_vocab(names)` 融合分类层。

#### `set_classes`（353–368）

禁止类名含空格；比较现有 names 与 classes；无 embeddings 则 `get_text_pe`；`model.set_classes`；必要时清 predictor。

#### `_prompt_embedding_model`（370–375）

stem 标识：优先 `pt_path`/`ckpt_path`，否则 yaml_file；`-seg` 后缀去掉。

#### `save_prompt_embeddings`（377–406）

校验 `pe` 形状 `(1, classes, dim)` 与 names 一致且 finite；后缀须 `.npz`；`np.savez_compressed` 存 embeddings/names/model 标识。

#### `load_prompt_embeddings`（408–438）

校验 npz 三键与 model 标量字符串、架构匹配、`float32` 形状与 head `embed` 维；finite 检查；`set_classes` 到当前设备。

#### `val`（440–466）

默认 `rect=not load_vp`；无 data 用 `_val_data()`；合并 overrides；实例化 validator；`validator(..., load_vp, refer_data)`；写 `self.metrics`。

#### `predict`（468–568）— 视觉提示分支

- 空 prompts → 仅设默认 `agnostic_nms=True` 走 `super().predict`。
- 非空：默认 `conf=0.05`；断言未 fused；须有 bboxes/cls；维度与 nested（每图一组）检查；4D numpy source 禁止；multi 时 prompts 与 source 数量一致；类索引为非字符串标量。
- 算 `num_cls`；构建 overrides（batch=1, predict, save=False）；选 `YOLOEVPSegPredictor` 或 `YOLOEVPDetectPredictor`。
- predictor 类型变则新建，否则更新 `PREDICTOR_SETUP_KEYS`。
- `set_prompts`；`setup_model`；临时 names `object{i}`。
- 无 refer_image 时视频/流取首帧为 refer；有 refer 则 `get_vpe` + `set_classes` + 可能改 task + `predictor=None`。
- 无 prompts 但 predictor 为 VP 类则清 predictor。
- 最后 `super().predict`。