# `.github/scripts/create-export-env.py`

该 CLI 根据 `ultralytics.engine.exporter.EXPORT_ENVS` 注册表，用 `uv` 为每个隔离导出环境创建独立虚拟环境、安装带 extras 的可编辑包与配方中的 torch/索引/requirements，并执行配方里的 smoke `yolo` 命令做冒烟验证。供 CI「IsolatedExports」作业与本地复现某导出格式依赖。`isolated-deepx` 在内存不足 15GiB 时自动跳过。`--list` 仅打印环境 ID 列表。

## 逐段说明

### `isolated_env_ids`（第 21–25 行）

返回所有配方中 `python` 字段非空的 env id，排序时将 `isolated-deepx` 置后。

### `build_env`（第 28–73 行）

删除旧 venv，`uv venv` 指定 Python 版本与 `--seed`。`uv pip install -e REPO[extras]`、pytest、可选 torch pin、requirements 与额外 index、`--torch-backend cpu`。若配方含 `env` 字典，写入 `sitecustomize.py` 做 `setdefault`。对每个 smoke 命令：首 token 为 `yolo` 时用 venv 内 yolo，否则 venv bin 下可执行文件；`YOLO_AUTOINSTALL=false` 与配方 env 合并，PATH 前置 venv 的 bin/Scripts。

### `main`（第 76–99 行）

解析 `--env`（可多次）与 `--list`；默认构建全部 isolated env；DeepX 内存门槛过滤；目标根目录默认 `/opt/venvs` 或 `ULTRALYTICS_ISOLATED_VENVS`。循环调用 `build_env`。