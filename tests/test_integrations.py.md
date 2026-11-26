# `tests/test_integrations.py`

本模块覆盖 Ultralytics 与可选第三方生态的集成：TensorBoard、Ray Tune、MLflow（含 run 生命周期）、NVIDIA Triton Docker 推理，以及 `faster-coco-eval` 加速的 COCO JSON 评估。多数用例在依赖未安装时 skip；训练类测试统一用小 `imagenet10`、`imgsz=32`、少 epoch、CPU 设备以降低 CI 成本。MLflow 测试使用临时 SQLite URI 与 monkeypatch 环境变量，避免污染用户全局 tracking。Triton 测试会 pull 官方容器、挂载临时 model repository，经 HTTP 对导出 ONNX 做远程 predict。

## 逐段说明

### `test_tensorboard`（第 16–21 行，`slow`）

临时打开 `SETTINGS["tensorboard"]`，对 `yolo26n-cls.yaml` 训练 3 epoch，finally 关闭 tensorboard 开关。

### `test_model_ray_tune`（第 24–29 行）

需 `ray`；调用 `YOLO.tune(use_ray=True, ...)` 单次迭代、短 grace period，验证 Ray 调参路径可运行。

### `test_mlflow`（第 32–44 行）

需 `mlflow`；SQLite tracking、`SETTINGS["mlflow"]=True`，训练后 `autolog(disable=True)` 与 `end_run()` 清理。

### `test_mlflow_keep_run_active`（第 47–74 行）

分三轮训练：环境变量 `MLFLOW_KEEP_RUN_ACTIVE=True` 时期望 `active_run()` 仍为 RUNNING；设为 False 或未设置时期望无活跃 run。每轮后按需 `end_run()`。

### `test_triton`（第 77–126 行）

需 `tritonclient`；动态 ONNX 导出、组装 Triton repo（`config.pbtxt` 空文件）、docker run 暴露 8000。轮询 `is_model_ready` 后 `YOLO(http://...)` predict `SOURCE`，最后 `docker kill` 容器。

### `test_faster_coco_eval`（第 129–156 行）

需 `faster-coco-eval`；对 detect/seg/pose 各跑 `*Validator`，`save_json=True`，设置 `is_coco` 与 annotations 路径后调用 `eval_json`，验证加速评估路径不抛错。