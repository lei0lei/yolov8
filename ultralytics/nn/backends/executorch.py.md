# `ultralytics/nn/backends/executorch.py`

本后端运行 **Meta ExecuTorch** 导出的 **`.pte`** 程序：可为单个 `.pte` 文件，或目录（内 `rglob("*.pte")`）。依赖经 `check_executorch_requirements()` 校验的 **`executorch.runtime.Runtime`**。加载后调用已导出方法名 **`forward`**。输入 **BCHW [0,1]**，以 **`torch.Tensor` CPU** 列表传入 `execute`；返回 ExecuTorch 原始输出列表，不在此转 NumPy。

## 逐段说明

### 模块导入（第 1–12 行）

`Path`、`torch`、`LOGGER`、`check_executorch_requirements`、`BaseBackend`。

### 类 `ExecuTorchBackend`（第 15–47 行）

#### `load_model`（第 22–36 行）

1. 日志。
2. `check_executorch_requirements()` 确保运行时可用。
3. `w = Path(weight)`：若 `w.is_dir()` 则 `next(w.rglob("*.pte"))`，否则直接用 `w` 作程序路径。
4. `Runtime.get().load_program(path)` → `program.load_method("forward")` 存 `self.model`。
5. `apply_metadata(read_metadata(w))`（文件或目录的旁路元数据）。

#### `forward`（第 38–47 行）

1. 输入 `im`：`self.model.execute([im.cpu()])`，单元素列表表示一批输入绑定。
2. 直接 `return` `execute` 的返回值（`list`），由基类/后处理按元数据解释各输出张量。本方法无布局变换、无量化、无多输出合并逻辑。