# `ultralytics/nn/backends/mnn.py`

本后端加载阿里巴巴 **MNN** 单文件 **`.mnn`**。依赖 PyPI 包 **`MNN`**，并通过 `import_execstack` 动态加载 `_mnncengine*.so`、`_tools*.so`。运行时用 `create_runtime_manager`（CPU、low precision、线程数约为 CPU 一半）与 `load_module_from_file(..., rearrange=True)`。输入 **BCHW [0,1]**，经 `MNN.expr.const` 以**主机内存指针**零拷贝绑定；输出为各输出变量的 `read().copy()` 列表（ARM 上必须 copy）。若图内嵌 NMS 且无输出张量，detect/pose 任务返回空检测占位。

## 逐段说明

### 模块导入（第 1–14 行）

`os`、`Path`、`numpy`、`torch`、`LOGGER`、`check_requirements`、`BaseBackend`。

### 类 `MNNBackend`（第 17–58 行）

#### `load_model`（第 24–40 行）

1. 日志；`check_requirements("MNN")`。
2. `MNN = import_execstack(...)` 加载原生引擎。
3. **`config`**：`precision=low`、`backend=CPU`、`numThread=(cpu_count+1)//2`。
4. `rt = create_runtime_manager((config,))`；`load_module_from_file(weight, [], [], runtime_manager=rt, rearrange=True)` → `self.net`；`self.expr = MNN.expr`。
5. `apply_metadata(read_metadata(weight))`。

#### `forward`（第 42–58 行）

1. `im = im.cpu().contiguous()`；`input_var = expr.const(im.data_ptr(), im.shape)`（MNN 读稠密主机缓冲）。
2. `output_var = self.net.onForward([input_var])`。
3. **`output_var` 非空**：`return [x.read().copy() for x in output_var]`，每个输出一个 numpy 副本。
4. **`output_var` 为空**：
   - 若 `metadata.args.nms` 且 `task in {detect, pose}`：返回 `[np.empty((batch, 0, 6))]`。
   - 否则 `RuntimeError("... no output tensors")`。

不在此做布局转换或反量化；元数据中的 NMS 标志与空输出分支配合 end2end 导出。