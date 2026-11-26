# `ultralytics/models/sam/model.py`

本模块在 Ultralytics 通用 `Model` 基类之上实现 SAM 系列的可提示分割用户接口。模块顶部 docstring 概括 Segment Anything 的能力：可提示分割、实时性、零样本迁移及 SA-1B 训练背景。`SAM` 类将任务固定为 `segment`，并根据权重文件名 stem 是否含 `sam2` / `sam3` 区分代际，从而在加载权重与选择 Predictor 时走不同分支。权重加载委托给 `build.build_sam` 或 `build_sam3.build_interactive_sam3`，避免在 `__init__` 中一次性拉起重型子图。`predict` / `__call__` 把框、点、标签提示打包进 `prompts` 字典，并强制 `retina_masks` 与 1024 推理尺度等默认覆盖项，与 YOLO 式 `predict` API 对齐。

## 逐段说明

### 模块 docstring 与导入（第 1–24 行）

- **第 1 行**：许可证头。
- **第 2–15 行**：模块说明文档，列举可提示分割、实时、零样本、SA-1B 等特性（仅供阅读，不参与逻辑）。
- **第 17 行**：`from __future__ import annotations` 启用推迟注解求值。
- **第 19–20 行**：`Path` 用于解析权重路径后缀与 stem；`Model` 为引擎层模型基类；`model_info` 用于统计参数量与层信息。
- **第 24 行**：从 `.predict` 导入 `Predictor`、`SAM2Predictor`、`SAM3Predictor`，供 `task_map` 属性按代际映射预测器类型（未导入视频/语义等专用 Predictor，那些由 `predict.py` 或其它入口直接使用）。

### 类 `SAM`（第 27–171 行）

继承 `ultralytics.engine.model.Model`，封装 SAM/SAM2/SAM3 权重路径校验、构建、推理与信息查询。

#### `SAM.__init__`（第 51–64 行）

- **第 60–61 行**：若 `model` 非空且 `Path(model).suffix` 不在 `{".pt", ".pth"}`，抛出 `NotImplementedError`，明确 SAM 推理仅支持 PyTorch  checkpoint 而非 YAML 结构。
- **第 62 行**：`self.is_sam2 = "sam2" in Path(model).stem`，按文件名 stem 子串判断是否为 SAM2（含 `sam2.1_*` 等）。
- **第 63 行**：`self.is_sam3 = "sam3" in Path(model).stem`，同理判断 SAM3。
- **第 64 行**：调用 `super().__init__(model=model, task="segment")`，触发基类中的权重下载/解析及 `_load` 钩子。

#### `SAM._load`（第 66–84 行）

- **参数**：`weights` 为 checkpoint 路径；`task` 可选，由基类传入，本实现未使用。
- **第 77–80 行**：若 `self.is_sam3` 为真，延迟导入 `build_interactive_sam3`，将返回值赋给 `self.model`（交互式 SAM3 跟踪/点框提示图）。
- **第 81–84 行**：否则延迟导入 `build_sam`，用其根据 checkpoint 名构建 SAM1/SAM2 并赋给 `self.model`。延迟导入减轻 `import ultralytics` 时无关 SAM 子模块的加载成本。

#### `SAM.predict`（第 86–110 行）

- **第 107 行**：`overrides` 固定 `conf=0.25`、`task=segment`、`mode=predict`、`imgsz=1024`，作为 SAM 默认推理配置。
- **第 108 行**：`kwargs = {**overrides, **kwargs, "retina_masks": True}`，用户 kwargs 可覆盖 overrides，但 `retina_masks` 始终为 `True`（后写字典项覆盖同键）。
- **第 109 行**：`prompts` 字典聚合 `bboxes`、`points`、`labels` 三个可选提示参数。
- **第 110 行**：调用 `super().predict(source, stream, prompts=prompts, **kwargs)`，由基类调度对应 Predictor 并返回 `Results` 列表或流式生成器。

#### `SAM.__call__`（第 112–135 行）

- 签名与 docstring 与 `predict` 对齐，`source` 默认为 `None`。
- **第 135 行**：直接 `return self.predict(...)`，把实例当可调用对象使用时行为与 `predict` 一致。

#### `SAM.info`（第 137–152 行）

- **第 152 行**：`return model_info(self.model, detailed=detailed, verbose=verbose)`，对已构建的 `nn.Module` 做层数/参数量/GFLOPs 等统计；`verbose=False` 时可能返回元组而不打印。

#### 属性 `SAM.task_map`（第 154–171 行）

- **第 169–171 行**：返回单键字典 `"segment" -> {"predictor": <类>}`。
- Predictor 类三元选择：`self.is_sam2` 为真则用 `SAM2Predictor`；否则若 `self.is_sam3` 则用 `SAM3Predictor`；否则用原版 `Predictor`。该映射供 `Model.predict` 在运行时实例化正确的分割预测器。