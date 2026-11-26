# `ultralytics/nn/distill_model.py`

本模块实现 YOLO 检测任务的知识蒸馏包装器：在同一训练批上，冻结教师、训练学生，并在 Detect 头之前的 neck 特征上施加分数加权的 L2 蒸馏损失。对外类型为 `FeatureHook`（可 pickle 的前向钩子）与 `DistillationModel`（`nn.Module` 包装器）。`engine/trainer.py` 在设置 `distill_model` 训练参数时构造 `DistillationModel`；`utils/torch_utils.py` 在保存/加载逻辑中识别该类；`nn/tasks.py` 的 `load_checkpoint` 若发现 checkpoint 内含蒸馏包装，会拆出 `student_model` 用于部署。对外行为上，`forward` 在推理时等价于学生的 `predict`，训练时 `forward(dict)` 走 `loss` 并返回检测损失与 `dis_loss` 拼接的张量。

## 导入说明（行 3–10）

- **PyTorch**：`F`、`nn` 用于投影 MLP 与 MSE 加权。
- **`Detect`**：用于 `get_distill_layers` 定位蒸馏层索引。
- **`copy_attr`**：从学生复制 `args`、`names` 等训练元数据到包装器。

## 逐段说明

### 类 `FeatureHook`（行 13–26）

#### `__init__(feat_dict, idx)`（行 16–19）

保存共享字典引用与特征层级键 `idx`（0…n-1 为 neck，n 为 Detect 头）。

#### `__call__(module, inputs, output)`（行 21–26）

前向钩子：将层输出写入 `feat_dict[idx]`。Detect 头输出可能是 tuple/dict，故不做类型收缩。

### 类 `DistillationModel`（行 29–353）

#### `__init__(teacher_model, student_model)`（行 60–117）

**类型与通道对齐（行 68–77）**：`teacher_model` 必须是 `nn.Module`；读取学生 `yaml.channels`（默认 3）；若教师输入通道不一致，则用教师类型以学生 YAML 重建教师并 `load` 原权重，使输入通道匹配。

**设备与冻结（行 77–79）**：教师迁移到学生参数所在设备，`_freeze_teacher()` 设 `eval` 且 `requires_grad=False`。

**蒸馏层（行 81–84）**：对学生、教师分别 `get_distill_layers`；层级数必须相等，否则 `ValueError`。

**钩子注册（行 86–91）**：空字典 `_teacher_feats`、`_student_feats`，列表存 hook 句柄，调用 `_register_feature_hooks()`。

**探针前向（行 93–102）**：`imgsz` 取自 `student_model.args.imgsz`；`eval`+`no_grad` 用零张量 `(2,ch,imgsz,imgsz)` 各跑一遍教师与学生，钩子填满特征字典；再 `student_model.train()`。从字典取出各层输出列表。

**投影器（行 104–117）**：`copy_attr` 复制学生属性到 `self`；`self.dis = student_model.args.dis`。对除最后一层（Detect 头）外的每一对 neck 特征，用 `decouple_outputs` 取张量维度，堆叠 `1x1 Conv → ReLU → 1x1 Conv` 将学生通道对齐到教师，组成 `nn.ModuleList` 并 `.to(device)`。

#### `__getstate__`（行 119–131）

序列化前 **就地** `clear()` 特征字典（钩子共享同一 dict 对象），复制 `__dict__` 并将 hook 列表置空，避免 pickle 带 `grad_fn` 的中间张量。

#### `__setstate__`（行 133–139）

老 checkpoint 无 `teacher_feats_idx` 时默认等于 `feats_idx`；重建空特征 dict；`_register_feature_hooks()` 恢复钩子。

#### `_remove_feature_hooks`（行 141–149）

移除学生与教师上已注册钩子并清空列表。

#### `_clear_feature_hooks(module)`（行 151–156）

遍历 `module._forward_hooks`，删除类型为 `FeatureHook` 的条目，防止重复注册。

#### `_register_feature_hooks`（行 158–170）

先 `_remove_feature_hooks`；对每个 `(s,t)` 层级对：清旧钩子，在学生 `model[s]`、教师 `model[t]` 上注册 `FeatureHook`，按层级 `i` 写入字典。

#### `get_distill_layers(model)`（行 172–191）

静态：遍历 `model.model`，找到 `Detect` 实例，返回 `[*m.f, m.i]`（各 neck 输入层索引 + 头自身索引）。无 Detect 则 `ValueError`。

#### `_freeze_teacher`（行 193–200）

教师为 `None` 则返回；否则 `eval` 并关闭全部参数梯度。

#### `train(mode=True)`（行 202–206）

`super().train(mode)` 后再次 `_freeze_teacher`，保证教师始终在 eval。

#### `forward(x, *args, **kwargs)`（行 208–221）

`x` 为 `dict` 时调用 `loss`（训练/验证）；否则 `student_model.predict(x, …)` 推理。

#### `fuse(verbose, imgsz)`（行 223–234）

移除钩子后返回 `student_model.fuse(...)`，丢弃蒸馏包装以便导出/推理。

#### `loss(batch, preds=None)`（行 236–283）

**验证分支（行 248–253）**：`not self.training` 时，若无 `preds` 则 `student_model(batch["img"])`；常规 `student_model.loss`；`dis_loss` 填零 detached；返回 `cat(regular_loss, loss_distill)` 与 `loss_items`。

**训练分支（行 255–283）**：清空特征 dict；`no_grad` 教师前向；学生前向得 `preds`；`regular_loss, loss_items = student_model.loss(batch, preds)`。

**蒸馏项（行 264–279）**：`n = len(feats_idx)-1`（neck 层数）。教师头特征经 `decouple_outputs` 取 `one2many`/`one2one` 的 `scores` 平均为 `teacher_scores`；按各 neck 特征图 `H*W` 在最后一维 `split`，每段 `sigmoid().max(dim=1, keepdim=True)` 得空间权重元组。

对每个 neck 层 `i`：教师/学生特征 `decouple_outputs` 后，学生经 `projector[i]`，累加 `loss_sl2(...) * self.dis`。

**收尾（行 281–283）**：记录 `dis_loss`；`loss_distill *= batch["img"].shape[0]`（批大小缩放）；与 `regular_loss` 拼接返回。

#### `loss_sl2(student_feat, teacher_feat, feat_idx, teacher_scores)`（行 285–305）

将 `(N,C,H,W)` reshape 为 `(N,C,HW)`；`mse_loss(..., reduction="none")`；用 `teacher_scores[feat_idx]` 加权求和，分母为 `teacher_score.sum() * c + 1e-9`。返回标量加权 MSE。

#### `criterion` 属性（行 307–315）

get/set 转发到 `student_model.criterion`。

#### `init_criterion`（行 317–319）

委托 `student_model.init_criterion()`。

#### `end2end` 属性（行 321–329）

读写学生 `end2end`，供 validator/predictor 控制 NMS-free 头。

#### `set_head_attr(**kwargs)`（行 331–333）

转发学生 `set_head_attr`。

#### `decouple_outputs(preds, branch="one2one")`（行 335–353）

统一 YOLO 训练/验证输出格式：tuple 取 `[1]`（验证输出）；dict 且含 `branch` 则取该分支；否则原样返回张量或 dict。