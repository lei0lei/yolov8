# `ultralytics/utils/export/axelera.py`

**Axelera AIPU**：在进程锁内量化并编译 PyTorch YOLO 为 **`.axm`** 与 `compiler_config_final.toml`，处理编译器对绝对路径与全局环境的限制。

## 逐段说明

### 模块级（第 20–25 行）

- 固定 `AXELERA_SDK="1.9.0"`；模块锁 `_AXELERA_EXPORT_LOCK` 串行化导出。

### `torch2axelera`（第 28–130 行）

1. **with 锁**：保存/恢复 PATH、protobuf 实现、`AXELERA_DEVICE_DIR`。
2. `check_requirements(axelera-devkit)`；设置 device 目录；`extract_ultralytics_metadata`。
3. 若 `output_dir` 存在则 rmtree。
4. `CompilerConfig`（0.25 资源、单核、batch multicore、`output_axm_format`）。
5. `compiler.quantize`（校准集 + `transform_fn`）。
6. **TemporaryDirectory** 相对路径 `compile_dir` 调用 `compiler.compile`（避免绝对路径 bug）。
7. 移动 `.axm` 与 toml 到 output_dir；删中间文件；metadata.yaml。
8. **finally** 恢复环境变量。
9. 返回 output_dir 字符串。