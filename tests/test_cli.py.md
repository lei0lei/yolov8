# `tests/test_cli.py`

本模块通过子进程调用 `yolo` 可执行文件，端到端验证 CLI 对全部任务的训练、验证、预测、导出与蒸馏。覆盖 help/checks/version/settings/cfg 等特殊子命令、设置文件 schema 迁移与 API key 过滤，以及导入时不应提前加载 torchvision/SAM3 几何模块。导出测试将权重复制到临时路径避免并行冲突；RT-DETR/FastSAM/MobileSAM/Solutions 等有平台或依赖条件跳过。双 GPU 训练标记为 slow。

## 逐段说明

### `run`（第 17–19 行）

`subprocess.run(cmd.split(), check=True)` 执行空格分隔的 yolo 命令字符串。

### `test_special_modes`（第 22–29 行）

依次运行 help、checks、version、settings reset、自定义 weights/datasets 目录、cfg，断言进程退出成功。

### `test_settings_migration`（第 32–55 行）

参数化 legacy 与 `ul_` 前缀 API key。写入旧版 `settings_version` 0.0.7 与已移除的 hub/neptune 键；加载后断言 `runs_dir` 保留、无效 api_key 清空、版本升至 0.0.8、废弃键不存在。

### `test_cli_imports_defer_torchvision`（第 58–67 行）

子进程执行 import YOLO 与 SAM Predictor，断言 `torchvision` 与 `sam3.geometry_encoders` 不在 `sys.modules`。

### `test_train` / `test_val` / `test_predict`（第 70–88 行）

对 `TASK_MODEL_DATA` 每项跑 CLI：`train` 1 epoch cache=disk（树莓派跳过 train）；`val`/`predict` 对 `nms` 为 None 与 False 各跑一次，含 agnostic_nms 或 save。

### `test_export`（第 91–99 行）

`MODELS` 中每个权重复制到 `tmp_path`，对 torchscript 导出两种 nms 模式。

### `test_distill`（第 102–113 行）

四任务各 1 epoch CLI 蒸馏，student 为 nano yaml、teacher 为对应 s 权重。

### `test_rtdetr`（第 116–125 行）

需 torch≥1.11；predict 与 train 命令含故意空格/逗号测试参数清洗。

### `test_fastsam`（第 128–160 行）

CLI val/predict 后 API 层测试 PIL 与路径源、点提示顺序对负样本掩膜的影响、`remove_small_regions`、联合 bbox/point/text 提示。

### `test_mobilesam`（第 163–190 行）

点/框/目录批处理与 mask prompt 浮点 logits 经 `_prepare_prompts` 保持原值。

### `test_train_gpu`（第 197–204 行，`slow`）

需 CUDA 且至少 2 卡；device=0 与 device=0,1 各训练 1 epoch。

### `test_solutions`（第 207–214 行）

十类 solutions 子命令，workout 用 pose 视频源，其余用 demo 视频，`verbose=False`。