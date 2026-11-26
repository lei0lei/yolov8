# `ultralytics/nn/tasks.py`

`tasks.py` 是 Ultralytics YOLO 家族的**模型定义与装配中心**：从 YAML 解析层图（`parse_model`）、实例化各任务 `*Model` 类、实现通用前向/融合/加载权重，并提供 checkpoint 安全加载与任务推断工具。对外公开的类型包括 `BaseModel` 及 detect/segment/semantic/depth/classify/pose/obb 等任务模型、`WorldModel`/`YOLOEModel` 开放词汇模型、`RTDETRDetectionModel`、`Ensemble`，以及 `load_checkpoint`、`torch_safe_load`、`parse_model`、`yaml_model_load`、`guess_model_task` 等函数。`models/yolo/model.py` 与 `engine/model.py` 按任务选用对应类；各 `models/yolo/*/train.py` 构造训练用模块；`nn/backends/pytorch.py` 调用 `load_checkpoint`；训练器在蒸馏时使用 `DistillationModel` 包装的学生定义亦源于此文件构建的 `DetectionModel`。本文件不实现具体卷积块（在 `nn/modules`）或损失公式细节（在 `utils/loss`），但负责任务类如何选择损失类并在 `forward(dict)` 时触发。

## 导入说明（行 3–119）

- **标准库**：`contextlib`、`importlib`、`pickle`、`re`、`threading`、`deepcopy`、`Path`——用于安全加载、临时模块别名、YAML 路径处理。
- **`numpy`、`torch`、`nn`**：张量运算与模块基类。
- **`autobackend.check_class_names`**：`YOLOEModel` 校验类名。
- **`nn.modules` 大批量符号**：`parse_model` 通过 `globals()[m]` 将 YAML 中的模块名字符串解析为类（Conv、C2f、Detect、RTDETRDecoder 等）。
- **`ultralytics.utils`**：`DEFAULT_CFG_DICT`、`LOGGER`、`SAFE_LOAD`、`YAML`、`IterableSimpleNamespace`、`colorstr`、`emojis` 等配置与日志。
- **`utils.checks`**：`check_file`、`check_yaml`、`REMOTE_FILE_PREFIXES`。
- **`utils.loss`**：各任务 `*Loss` / `E2ELoss`，供 `init_criterion` 返回。
- **`utils.ops.make_divisible`**：宽度缩放时通道对齐到 8 的倍数。
- **`utils.patches.torch_load`**：兼容各 PyTorch 版本的 `torch.load`。
- **`utils.torch_utils`**：融合 BN、初始化权重、`intersect_dicts`、`is_qat`、`model_info`、`restore_qat`、`scale_img`、`smart_inference_mode`、`time_sync` 等。

## 逐段说明

### 类 `BaseModel`（行 122–473）

所有 YOLO 任务模型的抽象基类，持有 `model`（`Sequential` 层列表）、`save`（需缓存输出的层索引）、`stride` 等。

#### `forward(x, *args, **kwargs)`（行 147–163）

若 `x` 为 `dict`（含 `img` 与标签），调用 `loss` 返回 `(loss_tensor, loss_items)`；否则 `predict` 走推理。这是训练循环与 validator 与 predictor 的统一入口。

#### `predict(x, profile, augment, embed)`（行 165–180）

`augment=True` 时走子类可覆盖的 `_predict_augment`；否则 `_predict_once`。`profile` 逐层计时；`embed` 指定层索引返回全局平均池化后的嵌入向量。

#### `_predict_once(x, profile, embed)`（行 182–208）

初始化输出列表 `y`、计时 `dt`、嵌入列表。将 `embed` 规范为 `frozenset` 层号，记录 `max_idx`。

**主循环（行 197–208）**：遍历 `self.model` 每层 `m`：

- **`m.f != -1`**：`f` 为 int 时从 `y[m.f]` 取单一路由；为 list 时组装 `[x if j==-1 else y[j] for j in m.f]`（多输入 concat/head）。
- **`profile`**：调用 `_profile_one_layer`。
- **`x = m(x)`** 执行层。
- **`y.append`**：若 `m.i in self.save` 保存输出，否则存 `None`。
- **嵌入**：对在 `embed` 中的层做 `adaptive_avg_pool2d` 展平；若当前层为 `max_idx`，拼接所有嵌入并 `unbind` 返回。

循环结束返回最后一层输出 `x`（检测头等原始输出）。

#### `_predict_augment(x)`（行 210–216）

默认警告不支持增强，回退 `_predict_once`。

#### `_profile_one_layer(m, x, dt)`（行 218–242）

可选 `thop` 计 FLOPs。对 Detect 最后一层且 `x` 为 list 时 copy 输入避免 inplace 问题。计时 10 次前向，记录毫秒与参数量；首层打印表头。

#### `fuse(verbose, imgsz)`（行 244–281）

QAT 时跳过 `Conv2/ConvTranspose/RepConv/RepVGGDW` 的某些融合。若 `not is_fused()`，遍历模块：Conv 系 `fuse_conv_and_bn` 并 `forward_fuse`；Rep 模块 reparam；`Detect.fuse()` 去掉无用分支。最后 `info()`。返回 `self`。

#### `is_fused()`（行 283–290）

检测是否仍存在带 `bn` 的 Conv、未融合的 Rep、或未 fuse 的双分支 Detect。

#### `info(detailed, verbose, imgsz)`（行 292–303）

委托 `model_info`。

#### `_apply(fn)`（行 305–325）

`super()._apply` 后对最后一层：若为 `Detect` 子类，对 `stride/anchors/strides` 应用 `fn`（`.to(device)` 等）；`RTDETRDecoder` 则处理 `anchors`、`valid_mask`。

#### `load(weights, verbose)`（行 327–381）

从 dict checkpoint 优先取 `ema` 再取 `model`；`state_dict` 转 FP32。

**检测头 cls 分支重建（行 342–359）**：当首尾均为未融合 `Detect` 且 `cv2` 输入通道一致时，对 `cv3`/`one2one_cv3` 若输出通道（类数）不同，deepcopy 源分支、替换最后一层 Conv 输出维度、继承 `requires_grad`，写回并 `bias_init()`。

**类名重映射（行 361–362）**：`_remap_cls_by_names` 按名称复制 cls 权重行。

**交集加载（行 364–381）**：`intersect_dicts` 与 `load_state_dict(strict=False)`。首层 conv 通道不匹配时 RGB→灰度求和规则填充。全部加载且非 World/YOLOE 时，用源 `names` 替换 YAML 占位名。记录 `pt_path` 与 transferred 日志；源已融合而当前未融合时警告。

#### `_remap_cls_by_names(csd, src_model, verbose)`（行 383–451）

比较源/目标 `names`；占位名 `{i:str(i)}` 跳过。建立目标类名到源索引的 `idx` 张量；无匹配或已同序则返回 0。

遍历 `named_modules` 找 `Detect` 或 `SemanticSegment`，收集 cls 相关 `weight/bias` 键（`cv3`、`one2one_cv3`、`proto.semseg`、`classifier` 等，且 `out_channels==tgt_nc`）。

对每个匹配键：形状除第 0 维外须一致；`v_tgt[valid]=v_src[idx[valid]]`；从 `csd` pop 防止重复错误顺序。返回 remapped 计数。

#### `loss(batch, preds)`（行 453–469）

懒初始化 `criterion=init_criterion()`；无 `preds` 时 `forward(batch["img"])`；调用 `criterion(preds, batch)`。

#### `init_criterion()`（行 471–473）

基类 `NotImplementedError`，子类必须实现。

---

### 函数 `_initialize_yolo_model(model, cfg, ch, nc, verbose)`（行 476–492）

统一 YOLO 检测系（及共享路径）初始化：`yaml` 为 dict 或经 `yaml_model_load`；Deprecated `Silence`→`nn.Identity` 警告并改写 YAML。写入 `channels=ch`；`nc` 参数覆盖 YAML `nc` 并打日志。`parse_model(deepcopy(yaml), ch, verbose)` 得到 `model.model` 与 `save`；`names` 默认为数字字符串；`inplace` 取自 YAML 默认 True。

---

### 类 `DetectionModel(BaseModel)`（行 495–653）

标准 YOLO 检测架构。

#### `__init__(cfg, ch, nc, verbose)`（行 523–561）

`super()` 后 `_initialize_yolo_model`。最后一层为 `Detect` 子类时：虚拟输入 `256×256`，`eval` 下局部 `forward` 取 `one2many.feats` 或等价输出，计算每层 stride `s/shape[-2]` 写入 `m.stride` 与 `self.stride`；`bias_init()` 一次。非 Detect 头（如 RT-DETR 前段）则 `stride=[32]`。`initialize_weights`；可选 `info()`。

#### 属性 `end2end`（行 563–572）

get：读 `model[-1].end2end`；set：若为 `Detect` 则写入头，用于 NMS-free 推理分支选择。

#### `set_head_attr(**kwargs)`（行 574–585）

对 `model[-1]` 存在属性则 `setattr`，否则警告。

#### `_predict_augment(x)`（行 587–609）

若 `end2end` 或最后一层非纯 `Detect`，警告并单尺度。否则三尺度 `s=[1,0.83,0.67]`、翻转 `f` 组合，`scale_img` 后 `super().predict` 取 `[0]`，`_descale_pred`、`_clip_augmented`，`cat` 在 anchor 维，返回 `(y, None)`。

#### `_descale_pred`（行 611–631）

box 前 4 维除以 scale；按 flip 类型反射 `y` 或 `x` 坐标。

#### `_clip_augmented(y)`（行 633–649）

按 `nl` 与网格数裁掉 TTA 产生的多余 anchor 尾段。

#### `init_criterion()`（行 651–653）

若存在 `one2one_cv2`（双分支 E2E 头）返回 `E2ELoss(self)`，否则 `v8DetectionLoss(self)`。

#### `forward`

继承 `BaseModel.forward`（dict→loss，张量→predict）。

---

### 类 `OBBModel(DetectionModel)`（行 656–685）

#### `__init__`（行 672–681）

仅 `super().__init__` 传参，架构由 OBB YAML 决定。

#### `init_criterion()`（行 683–685）

E2E 头时 `E2ELoss(self, v8OBBLoss)`，否则 `v8OBBLoss(self)`。

---

### 类 `SegmentationModel(DetectionModel)`（行 688–718）

#### `__init__`（行 704–713）

`super` 使用 seg YAML。

#### `init_criterion()`（行 715–718）

`Segment26` 头用 `SegmentLoss26`，否则 `v8SegmentationLoss`；E2E 时用 `E2ELoss` 包装。

---

### 类 `SemanticSegmentationModel(BaseModel)`（行 721–790）

语义分割，无检测框。

#### `__init__`（行 736–778）

`_initialize_yolo_model` 后，若尾层为 `SemanticSegment`：注册 forward hook 记录所有 4D 输出的最小 `H`，`stride = 256/min_h`（捕捉最深 backbone 步长，而非仅 head 输入）。`initialize_weights` 与 `info`。否则 `stride=32`。

#### `init_criterion()`（行 780–782）

`SemanticSegmentationLoss(self)`。

#### `_apply(fn)`（行 784–790）

额外对 `SemanticSegment.stride` 应用 `fn`。

#### `forward`

继承 `BaseModel`（无 augment 特化）。

---

### 类 `PoseModel(DetectionModel)`（行 793–833）

#### `__init__`（行 812–828）

非 dict 配置先 `yaml_model_load`；`data_kpt_shape` 可覆盖 YAML `kpt_shape` 并日志。`super` 后保存 `self.kpt_shape`。

#### `init_criterion()`（行 830–833）

`Pose26`→`PoseLoss26`，否则 `v8PoseLoss`；E2E 时 `E2ELoss` 包装。

---

### 类 `DepthModel(DetectionModel)`（行 836–860）

#### `__init__`（行 847–856）

标准检测式初始化（深度 YAML 含 `Depth` 头）。

#### `init_criterion()`（行 858–860）

`DepthLoss26(self)`。

---

### 类 `ClassificationModel(BaseModel)`（行 863–952）

#### `__init__`（行 887–897）

`super` 后 `_from_yaml`，不经过 `_initialize_yolo_model` 的 Detect stride 逻辑。

#### `_from_yaml(cfg, ch, nc, verbose)`（行 899–920）

加载 YAML；强制/覆盖 `nc`（缺失则 `ValueError`）；`parse_model`；`stride=Tensor([1])`；默认 `names`；`info()`。

#### `reshape_outputs(model, nc)`（行 922–948）

静态：找最后子模块，按 `Classify`/Linear/Sequential 中最后 Linear 或 Conv2d 替换输出维度为 `nc`。

#### `init_criterion()`（行 950–952）

`v8ClassificationLoss(self)`。

#### `forward`

继承 `BaseModel`。

---

### 类 `RTDETRDetectionModel(DetectionModel)`（行 955–1122）

Transformer 检测器，覆盖加载、损失与前向。

#### `__init__`（行 978–987）

`super` 完成 backbone+decoder 构建。

#### `_remap_cls_by_names`（行 989–1037）

覆盖：匹配 `score_head`/`class_embed` 张量行，适配 RT-DETR 类嵌入结构。

#### `init_criterion()`（行 1039–1043）

延迟导入 `RTDETRDetectionLoss(nc=self.nc, use_vfl=True)`。

#### `loss(batch, preds)`（行 1045–1091）

组 `targets`：`cls/bboxes/batch_idx/gt_groups`。`preds` 默认 `predict(img, batch=targets)`。训练/验证分支解码 `dec_bboxes/dec_scores` 与 denoising 分割；拼接 encoder 层输出；`criterion` 返回多分量 loss，对外 `sum(loss.values())` 与 detach 的 giou/cls/l1 三项。

#### `predict(x, profile, batch, augment, embed)`（行 1093–1122）

与 `_predict_once` 类似但**最后一层单独处理**：前面层按 `f` 路由；`head = model[-1]` 调用 `head([y[j] for j in head.f], batch)` 传入 GT 供训练去噪。中间层可提前返回 embed。

#### `forward`

继承 `BaseModel`（训练 batch 为 dict 时走上述 `loss`）。

---

### 类 `WorldModel(DetectionModel)`（行 1125–1253）

YOLO-World 开放词汇检测。

#### `__init__`（行 1149–1160）

预置 `txt_feats` 随机占位 `(1,nc,512)`、`clip_model=None`，再 `super` 构建网络。

#### `set_classes(text, batch, cache_clip_model)`（行 1162–1171）

`txt_feats = get_text_pe(...)`；更新 `model[-1].nc = len(text)`。

#### `get_text_pe(text, batch, cache_clip_model)`（行 1173–1194）

`build_text_model("clip:ViT-B/32")`；分批 `tokenize`+`encode_text`；reshape 为 `(-1, len(text), dim)`。

#### `predict(x, profile, txt_feats, augment, embed)`（行 1196–1235）

默认 `self.txt_feats`，cast 到 `x` dtype；按 batch 或 export 模式 `expand`。循环中对 `C2fAttn` 传 `(x,txt_feats)`；`WorldDetect` 传 `(x, ori_txt_feats)`；`ImagePoolingAttn` 更新 `txt_feats`；其余 `m(x)`。保存/embed 逻辑同 Base。

#### `loss(batch, preds)`（行 1237–1253）

初始化 criterion；`preds` 默认 `forward(img, txt_feats=batch["txt_feats"])`；`criterion(preds, batch)`。

#### `init_criterion`

继承 `DetectionModel.init_criterion`。

#### `forward`

继承 `BaseModel`（dict 时 `loss` 会用到 batch 内 `txt_feats`）。

---

### 类 `YOLOEModel(DetectionModel)`（行 1256–1529）

YOLOE 文本/视觉提示检测。

#### `__init__`（行 1283–1293）

`super` 后 `text_model` 取自 YAML 默认 `mobileclip:blt`。

#### `get_text_pe(text, batch, cache_clip_model, without_reprta)`（行 1296–1331）

`build_text_model(self.text_model)`；编码后 reshape 并转模型 `dtype`；若 `without_reprta` 直接返回；否则 `YOLOEDetect.get_tpe` 经文本辅助头。

#### `get_visual_pe(img, visual)`（行 1333–1344）

`self(img, vpe=visual, return_vpe=True)` 取视觉提示嵌入。

#### `set_vocab(vocab, names, one2one_vocab)`（行 1346–1385）

推理专用：校验 `names`；无梯度 warmup 一次前向；按 `end2one` 或双分支构建 `LRPCHead` ModuleList，删掉原 cv2/cv3 末层 Conv；可选 `head.fuse()`；更新 `nc` 与 `names`。

#### `get_vocab(names)`（行 1387–1412）

`get_text_pe`+`set_classes`；`head.fuse(self.pe)`；从 cv3 末层抽出 `vocab` ModuleList 返回。

#### `set_classes(names, embeddings)`（行 1414–1432）

禁止在 prompt-free（已有 `lrpc`）模型上调用；校验 3D embeddings 与融合状态；写 `names`、`pe`、`nc`。

#### `get_cls_pe(tpe, vpe)`（行 1434–1453）

拼接文本/视觉/默认 `pe` 沿类维 `cat`。

#### `predict(x, profile, tpe, augment, embed, vpe, return_vpe)`（行 1455–1497）

遇 `YOLOEDetect`：可选 `get_vpe`；`return_vpe` 时仅返回视觉嵌入；否则 `get_cls_pe(get_tpe(tpe), vpe)` 扩 batch 后 **append 到输入 list** `x.append(cls_pe)` 再 `m(x)`。其余层标准路由。

#### `loss(batch, preds)`（行 1499–1529）

若有 `visuals` 用 `TVPDetectLoss`（可 `E2ELoss` 包装），否则 `init_criterion()`。`forward` 传入 `tpe`/`vpe` 与 batch 字段一致。

#### `init_criterion`

继承 `DetectionModel`（标准闭集检测损失）。

#### `forward`

继承 `BaseModel`。

---

### 类 `YOLOESegModel(YOLOEModel, SegmentationModel)`（行 1532–1584）

多继承：YOLOE 提示 + 分割头损失。

#### `__init__`（行 1548–1557）

`super(YOLOEModel, self)` 解析链：`YOLOEModel.__init__` 构建。

#### `loss(batch, preds)`（行 1559–1584）

视觉提示时用 `TVPSegmentLoss`（可 E2E）；否则分割 `init_criterion`；`super().loss(batch, preds)` 调用 `YOLOEModel.loss` 逻辑。

#### `init_criterion`

分割父类规则（`SegmentLoss26` / `v8SegmentationLoss`）。

---

### 类 `Ensemble(ModuleList)`（行 1587–1625）

#### `__init__`（行 1605–1607）

空 ModuleList。

#### `forward(x, augment, profile)`（行 1609–1625）

对每个子模块 `module(x, augment, profile)[0]` 收集输出，`torch.cat(y, dim=2)` 在 anchor 维拼接（NMS ensemble 约定），返回 `(y, None)`。

---

### 函数 `temporary_modules(modules, attributes)`（行 1634–1693）

上下文管理器，线程锁内：按 `attributes` 映射在已导入模块上 `setattr` 别名；按 `modules` 映射重写 `sys.modules` 路径；`yield` 后逆序恢复。用于加载旧版 pickle 路径的 checkpoint。

---

### 类 `_SafeLoad`（行 1696–1858）

受限 `weights_only=True` 加载的注册表与工具。

#### 类属性（行 1705–1711）

`SUPPORTED` 检测 `get_unsafe_globals_in_checkpoint`；`_registry` 惰性构建；`_lock` 保护 `add_safe_globals`。

#### `register(weight)`（行 1714–1777）

读取 checkpoint 需要的全局名；首次 `_build()` 注册表；对 `ultralytics.nn.tasks/modules` 等模块按需 `importlib` 并加入 `nn.Module` 子类；torchvision transforms / `CLIP` 相关类按需追加；`numpy.dtype` 特殊项；`add_safe_globals(entries)`。

#### `activation(act)`（行 1779–1816）

用 `ast` 解析仅允许 `nn.Class(...)` 字面量形式，实例化激活模块供 YAML `activation` 字段使用。

#### `_build()`（行 1818–1858）

构造允许列表：numpy scalar、`IterableSimpleNamespace`、遗留类名别名（`Silence`→`Identity`、`YOLOv10DetectionModel`→`DetectionModel`）、受限 `getattr` pickle 钩子、跨平台 `pathlib` 路径类映射等。返回 name→class 字典。

---

### 函数 `torch_safe_load(weight, safe_only)`（行 1861–1994）

`safe_only` 默认 `SAFE_LOAD` 环境变量；不支持则降级。`check_suffix`+`attempt_download_asset`。

**内部 `_load()`（行 1891–1913）**：在 `temporary_modules` 别名下 `torch_load`；`safe_only` 时先 `_SafeLoad.register`。

**异常处理（行 1921–1981）**：

- 损坏/截断 zip 或 pickle：TorchScript 误装则明确错误；safe 模式 UnpicklingError 提示官方模型；仅官方资产 bare name 可删缓存重下。
- `ModuleNotFoundError`：区分 YOLOv5 旧仓、numpy 版本、缺失 `ultralytics.*` 子模块。

**收尾（行 1983–1993）**：拒绝 ScriptModule；非 dict 则包装为 `{"model": ckpt.model}` 并警告。

返回 `(ckpt, file)`。

---

### 函数 `load_checkpoint(weight, device, inplace, fuse)`（行 1997–2046）

远程 URL 经 `check_file` 下载。`torch_safe_load` 得 ckpt；`args` 合并默认 cfg 与 `train_args`。`candidate = ema or model` 须为 `Module`。

**蒸馏（行 2022–2024）**：若有 `student_model`，移除特征钩子后取学生子模块。

`float()`+可选 `restore_qat`；附加 `args`、`pt_path`、`task=guess_model_task`；缺 `stride` 则 32。`fuse`+`eval()`+`to(device)`。遍历模块设置 `inplace` 与 Upsample 兼容字段。返回 `(model, ckpt)`。

---

### 函数 `parse_model(d, ch, verbose)`（行 2049–2258）

**YAML 字典 `d` 已在外部由 `yaml_model_load` 读入**；本函数只消费内存中的 dict（键含 `backbone`、`head`、`nc`、`scales` 等）。

#### 第一段：全局超参与 scale（行 2063–2080）

读取 `nc, activation, scales, end2end, reg_max(默认16), depth_multiple, width_multiple, kpt_shape, scale`。若存在 `scales` 且未传 `scale`，取首 key 并警告；从表得到该型号的 `depth, width, max_channels`。保存 `Conv.default_act`，若 YAML 有 `activation` 则 `_SafeLoad.activation` 设为全局默认并打印。

#### 第二段：循环状态（行 2082–2085）

`ch = [ch]` 为通道栈（索引为层号，值为该层输出通道）；`layers, save, c2` 初始化；`legacy=True` 标记 v8 风格路由。

#### 第三段：`base_modules` 与 `repeat_modules`（行 2086–2142）

`base_modules`：需要 `c1,c2` 注入的卷积/块类型集合。`repeat_modules`：YAML 中 `n>1` 表示重复次数、构造时把 `n` 插入 args 的类型（C2f、C3 等）。

#### 第四段：主循环 `for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"])`（行 2143–2257）

**解析模块名 `m`（行 2144–2152）**：

- `nn.*` → `getattr(torch.nn, 后缀)`
- `torchvision.ops.*` → torchvision.ops
- 否则 → `globals()[m]`（本文件导入的 Ultralytics 模块类）

非 `nn.Module` 子类则 `TypeError`。

**解析 `args` 字符串（行 2153–2156）**：字符串参数尝试 `locals()` 名（如 `nc`）或 `ast.literal_eval`。

**深度倍数 `n`（行 2157）**：`n>1` 时 `n_ = max(round(n*depth),1)`，否则保持。

**分支 A — `m in base_modules`（行 2158–2185）**：

- `c1, c2 = ch[f], args[0]`：`f` 为来源层索引，输入通道来自 `ch[f]`，输出基准为 YAML args[0]。
- 非 `Classify`：`c2 = make_divisible(min(c2,max_channels)*width, 8)` 宽度缩放。
- `C2fAttn`：缩放 embed 维与 heads，校验 hidden 整除 `nh`。
- 组装 `args = [c1, c2, *args[1:]]`。
- `repeat_modules`：`args.insert(2, n)` 且构造次数 `n=1`（重复进模块内部）。
- `C3k2`/`A2C2f`/`C2fCIB`：按 `scale` 设 `legacy=False` 并追加 M/L/X 特有标志。

**分支 B — 其它模块类型（行 2186–2241）**（节选逻辑）：

- `AIFI`：`[ch[f], *args]`
- `HGStem/HGBlock`：三通道参数 + HGBlock 重复 `n`
- `ResNetLayer`：按 args[3] 决定是否 ×4 通道
- `BatchNorm2d`：`args=[ch[f]]`
- `Concat`：`c2 = sum(ch[x] for x in f)` 多输入求和通道
- **检测/分割/姿态/OBB 头**：`args.extend([reg_max, end2end, [ch[x] for x in f]])`；分割 proto 通道宽度缩放；对标准 Detect 系列设 `m.legacy=legacy`
- `Depth`：`[c_mid, 多尺度 ch 列表]`
- `SemanticSegment`：append 特征通道列表
- `v10Detect`、`ImagePoolingAttn`、`RTDETRDecoder`：插入多尺度 `ch` 列表到指定位置
- `CBLinear/CBFuse/TorchVision/Index`：各自 c1/c2 规则
- **默认**：`c2 = ch[f]`

**实例化与元数据（行 2243–2256）**：

- `n>1` 时 `Sequential(*(m(*args) for _ in range(n)))`，否则 `m(*args)`。
- 遗留 `SPPF` 补 `cv1.act`。
- 附加 `m_.i, m_.f, m_.type, m_.np`；verbose 打印表格行。
- `save`：从 `f`（int 或 list）收集需缓存的层索引 `% i`。
- `ch`：首层后清空为仅追踪输出通道列表，每层 `append(c2)`。

**返回（行 2257–2258）**：恢复 `Conv.default_act`；`Sequential(*layers), sorted(save)`。

---

### 函数 `yaml_model_load(path)`（行 2261–2281）

P6 旧命名 `yolov8n6`→`yolov8n-p6` 警告并改 stem；尝试 `check_yaml(path)` 或去掉 scale 字母的 unified 路径；`YAML.load`；`d["scale"]=guess_model_scale(path)`；记录 `yaml_file` 原路径字符串。

---

### 函数 `guess_model_scale(model_path)`（行 2284–2296）

正则 `yolo(e-)?[v]?\d+([nslmx])` 提取 n/s/m/l/x；失败返回 `""`。

---

### 函数 `guess_model_task(model)`（行 2299–2384）

**内部 `cfg2task(cfg)`（行 2309–2325）**：读 `head[-1]` 模块名字符串判断 classify/detect/semantic/segment/pose/obb/depth。

**输入为 dict（行 2328–2330）**：try `cfg2task`。

**输入为 Module（行 2332–2353）**：先尝试嵌套 `args["task"]`；再嵌套 `yaml` cfg2task；再扫 `modules()` 中 `SemanticSegment`、`Segment`、`Classify`、`Pose`、`OBB`、`Depth`、`Detect` 等类型。

**输入为路径（行 2355–2376）**：`BaseBackend.read_metadata` 的 `task`；否则按 stem/parts 中 `-seg`、`-cls` 等关键字。

**默认（行 2378–2384）**：警告并返回 `"detect"`。