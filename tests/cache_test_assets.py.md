# `tests/cache_test_assets.py`

本脚本在 CI「Assets」作业或本地首次并行 pytest 前，一次性预下载测试与基准所需的全部权重、数据集、Solutions 媒体与 CLIP 文本编码器。目的是让后续 xdist worker 只读已有文件，避免同时触发 GitHub Releases 下载竞态。权重列表覆盖各任务默认模型、蒸馏/分割/跟踪专用 checkpoint 及校准数据对应的 yaml。数据集分支对 ImageNet 类任务走 `check_cls_dataset`，其余走 `check_det_dataset(autodownload=True)`，并补全 COCO 标注 JSON。

## 逐段说明

### 模块级列表 `WEIGHTS` / `DATASETS`（第 22–45 行）

声明需缓存的权重文件名超集与各任务数据配置；与 `TASK2MODEL`、`TASK2DATA`、`TASK2CALIBRATIONDATA` 并集后还包含灰度、多光谱、格式测试等专用 yaml。

### `cache_weights`（第 48–56 行）

遍历 `WEIGHTS` 调用 `attempt_download_asset`；在 `MODEL.parent`（含空格路径）为每个任务权重做 `copy2`，保证 CLI/基准引用的路径存在。记录日志起止。

### `cache_datasets`（第 59–69 行）

按名称前缀选择分类或检测数据集检查器触发下载/解压；额外下载 COCO `instances_val2017.json` 与 `person_keypoints_val2017.json` 到 `DATASETS_DIR/annotations`。

### `cache_solution_assets`（第 72–81 行）

在 `solution_assets` 目录为 `SOLUTION_ASSETS` 每个值执行 `safe_download`（已存在则跳过）。

### `cache_clip_model`（第 84–94 行）

树莓派与特定 ARM64 Python 3.8 环境跳过。否则在 CPU 上实例化 `CLIP("ViT-B/32")` 触发权重缓存，随即删除对象释放内存。

### `main`（第 97–103 行）

顺序调用四个 cache 函数并打印完成日志；`__main__` 入口供 CI 直接 `python tests/cache_test_assets.py` 执行。