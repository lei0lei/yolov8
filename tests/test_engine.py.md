# `tests/test_engine.py`

本文件在 Python API 层测试 Ultralytics 引擎：导出回调、各任务 Trainer/Validator/Predictor 三联、语义 polygon 独立验证背景类、checkpoint 恢复与蒸馏恢复、灰度蒸馏通道对齐、非法 checkpoint 拒绝、NaN 与 fp16 溢出/非有限 EMA 的保存与消毒、多数据集自定义 Trainer 指标键，以及 `.pt` 模型 `pretrained` 参数语义。大量使用临时目录与小 `imgsz` 控制耗时；树莓派跳过完整训练矩阵。

## 逐段说明

### `test_func`（第 24–26 行）

打印桩回调，供各引擎注册测试使用。

### `test_export`（第 29–36 行）

`Exporter` 注册 `on_export_start` 回调；对 yaml 模型默认导出并 `YOLO(f)(SOURCE)` 推理。

### `test_task`（第 39–125 行）

参数化七任务（含 depth/semantic）。训练注册 `on_train_start` 并 `train()`；Validator 用 `get_cfg` 跑 `val(model=trainer.best)`；Predictor 设 64 方形 imgsz 并对 ASSETS 预测，detect 额外在空 `argv` 下预测。检测 yaml 路径断言有结果。`resume=trainer.last` 在仅训 1 epoch 后应 `AssertionError`（不可再 resume 已完成 run）。

### `test_semantic_polygon_val_background`（第 128–141 行）

独立语义验证器加载 coco8-seg，手工设 stride/device；dataloader 样本应有 `bg_class_idx==80` 且 semantic_mask 含 80 背景像素。

### `test_resume_incomplete`（第 144–177 行）

各任务训练 2 epoch 但在 epoch0 末 stop；checkpoint `epoch==0`；`resume=True` 应从 epoch 1 继续。

### `test_resume_invalid_checkpoint_args`（第 180–186 行）

篡改 checkpoint 中 `epochs=0`，resume 应抛 `ValueError` 含 epochs。

### `test_distill_resume`（第 189–229 行）

蒸馏训 1 epoch 中断；checkpoint 中 EMA 为 `DistillationModel`、teacher 已剥离、projector 保留；resume 后重建 teacher、`start_epoch==epoch==1`。

### `test_distill_grayscale`（第 232–242 行）

1 通道 student、3 通道 teacher；`DistillationModel` 应把 teacher yaml channels 同步为 1。

### `test_load_checkpoint_state_dict_rejected`（第 245–258 行）

三种非标准 `.pt` 结构应 `TypeError` 提示非 Ultralytics checkpoint。

### `test_nan_recovery`（第 261–275 行）

epoch1 注入 `loss=nan`；训练应完成且注入发生。

### `test_checkpoint_fp16_overflow`（第 278–299 行）

EMA 权重设为 1e5；仍保存 checkpoint 且 state 全有限；活 EMA 保持 fp32 原值。

### `test_checkpoint_nonfinite_ema_resync`（第 302–318 行）

EMA 参数设 inf；仍产出有限 checkpoint。

### `test_checkpoint_nonfinite_ema_and_model_sanitized`（第 321–338 行）

EMA 与 model 同参数非有限；checkpoint 仍保存且有限。

### `test_train_multi_custom_trainer_metrics_and_failure_keys`（第 341–370 行）

FakeTrainer 第二次失败；`data` 列表重复 yaml；结果字典键为 `coco8` 与 `coco8-2`，后者 None。

### `test_setup_model_respects_pretrained_arg_for_pt_models`（第 373–396 行）

参数化 `pretrained` True/False/路径；`setup_model` 应用 checkpoint yaml，且仅 True/路径时向 `get_model` 传入权重对象。