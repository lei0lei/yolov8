# `ultralytics/engine/model.py`

本模块定义任务无关的 `Model` 基类，统一 YOLO 系模型的训练、验证、预测、导出、跟踪、基准测试与超参搜索入口。对外接口为 `Model` 实例方法（`train`、`val`、`predict`、`export`、`track`、`benchmark`、`tune`、`calibrate` 等）及属性（`names`、`device`、`transforms`）。`ultralytics.models.yolo.model.YOLO` 等子类继承并实现 `task_map`，由 `_smart_load` 按 `task` 解析对应的 trainer、validator、predictor、model 类。CLI `yolo`、Jupyter 中的 `YOLO(...)`、以及 `benchmark()` 均经此层调度；导出格式（ONNX 等）以路径字符串持有 `self.model`，仅 `predict`/`val` 可用 PyTorch 专属方法会调用 `_check_is_pytorch_model` 拦截。

## 逐段说明

### 第 3～31 行：导入

- **标准库**：`__future__.annotations` 延迟注解；`ast` 解析导出元数据中的 `imgsz`/`names`；`inspect` 在 `_smart_load` 失败时取调用方函数名；`Iterator` 标注 `predict`/`embed` 流式返回；`Path` 路径；`Any` 泛型。
- **第三方**：`numpy`、`torch`、`PIL.Image` 供 `predict` 源类型与模块基类。
- **ultralytics.cfg**：`QUANTIZE_ALIASES` 规范化 `quantize`；`TASK2DATA` 默认数据集；`_handle_deprecation` 兼容旧参数；`get_cfg` 合并配置。
- **engine.results**：`Results` 预测结果类型。
- **nn**：`BaseBackend` 读导出元数据；`BaseModel`、`guess_model_task`、`load_checkpoint`、`yaml_model_load` 建图与加载。
- **utils**：日志、配置、回调、检查、分布式 `RANK` 等；`unwrap_model` 剥 DDP/compile 包装。

### 第 33～34 行：`PREDICTOR_SETUP_KEYS`

元组列出改变 predictor/tracker 装配的键：`device`、`dnn`、`data`、`nms`、`compile`、`channels_last`、`quantize`、`tracker`。`predict()` 中若这些键与当前 `predictor.args` 不一致则重建 predictor，避免陈旧后端或 NMS 设置。

### 第 37～88 行：类 `Model` 文档字符串

说明属性（callbacks、predictor、model、trainer、ckpt、cfg、overrides、metrics、task 等）与公开方法列表，供用户与文档生成器阅读，无运行逻辑。

### 第 90～143 行：`Model.__init__`

- 若 `model` 已是 `Model`：浅拷贝 `__dict__` 后 `return`，实现克隆式构造。
- 否则 `super().__init__()`，初始化回调默认表、predictor/trainer 为 `None`、空 `ckpt`、空 `overrides`、空 `metrics`、`task`、清空 `model_name`，`model` 转字符串并 `strip`。
- **Triton**：`is_triton_model` 为真则 `model_name`/`model` 设为 URL 字符串，`overrides["task"]` 默认 `detect`，直接返回（无本地权重）。
- **本地**：设 `CUBLAS_WORKSPACE_CONFIG` 抑制确定性警告；`.yaml/.yml` 走 `_new`，否则 `_load`。
- 最后 `del self.training`，避免与内层 `self.model.training` 属性冲突。

### 第 145～173 行：`Model.__call__`

将任意 `source`、`stream` 与 `**kwargs` 转调 `self.predict`，使实例可像函数一样推理。

### 第 175～197 行：`Model.is_triton_model`（静态）

懒导入 `urlsplit`；解析 URL，当同时有 `netloc`、`path` 且 `scheme` 为 `http` 或 `grpc` 时返回 `True`，用于识别远程 Triton 模型名。

### 第 199～228 行：`Model._new`

`yaml_model_load` 得 `cfg_dict`；保存 `self.cfg`；`task` 为参数或 `guess_model_task`；`_smart_load("model")(cfg_dict, verbose=...)` 构建 `nn.Module`；`overrides` 写入 `model`、`task`；`self.model.args` 合并 `DEFAULT_CFG_DICT` 与 overrides；`model.task`、`model_name` 设好。

### 第 230～268 行：`Model._load`

- `normalize_platform_uri` 得 `model_uri`；远程前缀则 `check_file` 下载到 `weights_dir`。
- `check_model_file_from_stem` 补 `.pt` 等后缀。
- **`.pt`**：`load_checkpoint` → `self.model`、`self.ckpt`；`task` 来自 checkpoint；`overrides`/`model.args` 经 `_reset_ckpt_args` 过滤；`ckpt_path` 为 `pt_path`。
- **非 `.pt`**：`check_file`；`model` 为路径字符串、`ckpt=None`；`task` 猜测；从元数据读 `imgsz` 写入 `overrides["imgsz"]`（取 `max`）。
- `overrides["model"]` 为 `ul://` URI 或本地路径；`task`、`model_name` 更新。

### 第 270～295 行：`Model._check_is_pytorch_model`

判断 `self.model` 为 `nn.Module` 或 `.pt` 路径字符串；否则 `TypeError`，提示仅 PyTorch 可 train/export 等，导出格式仅 predict/val。

### 第 297～322 行：`Model.reset_weights`

检查后遍历 `modules()` 调用 `reset_parameters`（若有）；全部 `requires_grad=True`；`overrides["pretrained"]=False`；`predictor=None`；返回 `self`。

### 第 324～354 行：`Model.load`

路径则 `load_checkpoint` 得权重与 `ckpt_path`；对象则 `ckpt={"model": self.model}`。`pop("pretrained")`；`self.model.load(weights)`；更新 `ckpt`、`ckpt_path`；清空 predictor；返回 `self`。

### 第 356～397 行：`Model.save`

懒导入版本、时间、`qat_state`/`strip_qat`。若为 Module：`deepcopy`、可选 `half`、连续内存；`strip_qat`。`updates` 含 `ema=None`、model、`modelopt`、`date`、`version`、`license`、`docs`；`torch.save({**self.ckpt, **updates}, filename)`。

### 第 399～420 行：`Model.info`

检查后委托 `self.model.info(detailed, verbose, imgsz)`，返回层数/参数量/GFLOPs 或 `None`。

### 第 422～451 行：`Model.fuse`

委托 `self.model.fuse`；`predictor=None`（融合改变计算图）；返回 `self`。

### 第 453～492 行：`Model.embed`

检查 PyTorch；`unwrap_model` 后须为 `BaseModel`，否则 `TypeError`。未传 `embed` 时默认 `[len(model.model)-2]`（倒数第二层）。转调 `predict(..., embed=...)`，返回嵌入张量而非 `Results`。

### 第 494～569 行：`Model.predict`（第一部分：参数与 predictor 生命周期）

- `source is None`：OBB 用 boats.jpg，否则 `ASSETS`，并警告。
- `is_cli`：`ARGV[0]` 为 yolo/ultralytics 且含 predict/track 相关子串。
- `custom` 默认 `conf=0.25`、`batch=1`、`save=is_cli`、`mode=predict`、`rect=True`、`embed=None`。
- `_handle_deprecation(kwargs)`；弹出 `prompts`（SAM）。
- `args` 合并 overrides、custom、kwargs；`quantize` 经 `QUANTIZE_ALIASES` 规范化。
- **重建 predictor 条件**：无 predictor、传入不同 predictor 类、或 `PREDICTOR_SETUP_KEYS` 任一在 args 中与当前不一致。
  - 则 `_smart_load("predictor")(overrides=args, ...)`，`setup_model(model=self.model, verbose=is_cli)`，赋 `self.predictor`。
  - 否则在锁外合并 `base_args` + `get_cfg`，`show` 再经 `check_imshow`。
- `cfg.prompts`；有 `set_prompts` 则调用。
- CLI 走 `predict_cli`（消费生成器不占内存），否则 `current(source, stream, cfg=cfg)`。

### 第 571～612 行：`Model.track`

`register_tracker(self, persist)` 注册跟踪回调；默认 `conf=0.1`、`batch=1`、`mode=track`；转调 `predict`。

### 第 614～648 行：`Model.val`

`custom={"rect": True}`；无 `data` 则 `_val_data()`。合并 args、`mode=val`。实例化 validator，`validator(model=self.model)`；`self.metrics=validator.metrics` 并返回。

### 第 650～693 行：`Model.calibrate`

仅 depth 任务、须 PyTorch、须有带 `cal_a`/`cal_b` 的 Depth 头。跑 validator；`fit_calibration_selective` 拟合尺度；成功则日志提示 `save`；返回 `(a,b)` 或 `None`。

### 第 695～753 行：`Model.benchmark`

检查 PyTorch；`benchmark` 与 `export_formats()`。`custom` 含 `verbose=False`、`nms=None`；`_handle_deprecation`；`mode=benchmark`。按 `format` 筛 `export_kwargs`（排除 batch/data/quantize/nms）。调用 `benchmark(model=self, data, imgsz, device, verbose, format, quantize, nms, **export_kwargs)`，返回 Polars DataFrame。

### 第 755～808 行：`Model.export`

检查 PyTorch；`Exporter` + `export_formats`。`custom` 从 `model.args["imgsz"]`、batch=1、重置 device 等。`Exporter(overrides=args)(model=self.model)` 返回路径。异常时除 ST Edge AI 外提示 Platform URL 后重抛异常。

### 第 810～916 行：`Model.train`（分段）

**配置合并（852～866）**：检查 PyTorch；`check_pip_update_available`。若 `kwargs["cfg"]` 存在则从 YAML 载入为 `overrides`。`custom` 解析 `data`（cfg 内、默认、resume 字符串、或 `TASK2DATA[task]`）、`model`、`task`；`args` 右端优先；`pretrained` 默认 True。

**多数据集（868～879）**：`data` 为 list/tuple 时用 `MultiTrainer`，可选自定义 trainer；`train()` 返回 `metrics`。

**resume=True（880～889）**：可恢复 checkpoint 则 `resume=ckpt_path` 并补 `data`；否则警告并 `resume=False`。

**Trainer 构造与权重（891～900）**：`(trainer or _smart_load("trainer"))(overrides=args)`。非 resume 且有 ckpt：`weights` 为 `None`（从零）或 `self.model` 或 `load_checkpoint(pretrained)`；`trainer.get_model(weights, cfg=self.model.yaml)`；同步 `self.model`；清空 predictor。

**训练后更新（901～916）**：`trainer.train()`；主进程 `RANK in {-1,0}` 加载 `best` 或 `last`，不存在则 `FileNotFoundError`；`load_checkpoint` 更新 model/ckpt；`_reset_ckpt_args` + `overrides["model"]=str(ckpt)`；`metrics` 来自 validator 或 ckpt `train_metrics`（DDP）。返回 `self.metrics`。

### 第 918～969 行：`Model.tune`

检查 PyTorch；默认 `optimizer=AdamW`（若 space 未指定）。`use_ray`：`run_ray_tune(self, *args, iterations, **kwargs)` 返回 ResultGrid。否则 positional `args` 禁止；`Tuner(args, model, callbacks)(iterations)`。

### 第 971～1002 行：`Model._apply`

检查 PyTorch。包装 `fn`：张量若被转换则 `predictor=None`；`super()._apply(apply)`；`overrides["device"]=self.device`；返回 `self`（支持 `.cuda()`/`.cpu()` 等）。

### 第 1004～1035 行：属性 `names`

`check_class_names`。Module 有 `names` 直接返回。无 predictor 时尝试元数据 `names`（`literal_eval`）。否则临时建 predictor、`setup_model(verbose=False)`，读 `predictor.model.names`。

### 第 1037～1056 行：属性 `device`

Module 则 `next(parameters()).device`，否则 `None`。

### 第 1058～1077 行：属性 `transforms`

有 `model.transforms` 则返回，否则 `None`。

### 第 1079～1099 行：`add_callback`

`self.callbacks[event].append(func)`。

### 第 1101～1126 行：`clear_callback`

`self.callbacks[event]=[]`。

### 第 1128～1148 行：`reset_callbacks`

每个 event 恢复为 `default_callbacks[event][0]` 单元素列表。

### 第 1150～1171 行：`_reset_ckpt_args`（静态）

仅保留 `imgsz`、`data`、`task`、`single_cls` 键值，丢弃训练专用参数，供加载后 overrides 干净。

### 第 1173～1187 行：`_val_data`

`overrides["data"]` 若 `check_data_portable` 则直接返回。非 str/Path 置 `None`；默认 `TASK2DATA[task]` 并警告缺失或找不到；返回 default。

### 第 1189～1215 行：`_smart_load`

`return self.task_map[self.task][key]`；异常包装为 `NotImplementedError`，含类名、调用函数名、`task`。

### 第 1217～1240 行：属性 `task_map`

基类 `raise NotImplementedError`，子类（如 `YOLO`）返回 detect/segment/... 到 model/trainer/validator/predictor 的映射。

### 第 1242～1258 行：`eval`

`self.model.eval()`；返回 `self`。

### 第 1260～1281 行：`__getattr__`

`name=="model"` 返回 `_modules["model"]`；否则 `getattr(self.model, name)`，使 `model.stride` 等透传到内层网络。