# `ultralytics/utils/dist.py`

本模块支撑 Ultralytics 单机多卡 DDP：为子进程挑选稳定本地端口、序列化 trainer 状态到临时 `.pt`、生成可被 `torch.distributed.run`/`launch` 执行的临时 `.py` 启动脚本，并在训练后清理临时文件。不直接启动进程，由 `generate_ddp_command` 拼出完整命令行供上层 `subprocess` 调用。依赖 `USER_CONFIG_DIR/DDP` 存放临时产物，并用 `cloudpickle` 保存 trainer 类、参数、模型与回调。

## 逐段说明

### `find_free_network_port`（第 20～48 行）

- **第 34～35 行**：局部 import `random`、`socket`。
- **第 39 行**：用 `SystemRandom().sample(range(10000, 32768), 10)` 取 10 个候选端口（避免与全局 RNG 同步导致多 DDP 进程选同一批口）。
- **第 40～45 行**：对每个端口建 TCP socket，`bind(("127.0.0.1", port))` 成功则返回；`OSError` 则继续下一个。
- **第 46～48 行**：均占用则 `bind(0)` 取系统分配端口（可能为 ephemeral，文档说明风险）。

### `generate_ddp_file`（第 51～113 行）

- **第 69 行**：`check_requirements("cloudpickle>=3.1.1")`。
- **第 72 行**：import `cloudpickle`。
- **第 74 行**：确保 `USER_CONFIG_DIR/DDP` 目录存在。
- **第 75～82 行**：`NamedTemporaryFile` 写模式，`delete=False`，后缀含 `id(trainer)` 便于后续识别。
- **第 83 行**：伴随状态路径为同名 `.pt`。
- **第 84～93 行**：`torch_save` 字典含 trainer 类型、`vars(trainer.args)`、`trainer.model`、`trainer.callbacks`，`pickle_module=cloudpickle`。
- **第 94～112 行**：向 `.py` 写入模板：`sys.path` 还原、`torch_load` 状态、`DEFAULT_CFG_DICT` 合并并清空 `save_dir`、`trainer(...)` 重建并 `train()`。
- **第 113 行**：返回临时 `.py` 路径字符串。

### `generate_ddp_command`（第 116～144 行）

- **第 129～130 行**：非 resume 时 `shutil.rmtree(trainer.save_dir)` 清旧运行目录。
- **第 131 行**：`file = generate_ddp_file(trainer)`。
- **第 132 行**：`TORCH_1_9` 决定用 `torch.distributed.run` 还是 `launch`。
- **第 133 行**：`find_free_network_port()` 得 `master_port`。
- **第 134～143 行**：命令为 `[python, -m, dist_cmd, --nproc_per_node, world_size, --master_port, port, file]`。
- **第 144 行**：返回 `(cmd, file)` 元组。

### `ddp_cleanup`（第 147～166 行）

- **第 164 行**：仅当路径含 `f"{id(trainer)}.py"` 判定为本次生成的临时脚本。
- **第 165 行**：`os.remove(file)` 删 `.py`。
- **第 166 行**：`Path(file).with_suffix(".pt").unlink(missing_ok=True)` 删状态文件。