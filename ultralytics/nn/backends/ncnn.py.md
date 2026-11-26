# `ultralytics/nn/backends/ncnn.py`

本后端加载腾讯 **NCNN** 导出：**`.param` + 同名 `.bin`**，或传入含 `*.param` 的目录。依赖 **`ncnn`** Python 绑定（`--no-deps` 安装）。可选 **Vulkan**：`device` 字符串以 `vulkan:` 开头时启用 `use_vulkan_compute` 并 `set_vulkan_device`；逻辑设备仍视为 CPU 张量输入。Linux 上按主机核心数调整 `num_threads`。输入 **BCHW [0,1] float** 逐张 `pyncnn.Mat` 送入；输出按**排序后的 output 名**抽取，多 batch 时 `np.stack` 拼回。

## 逐段说明

### 模块导入（第 1–13 行）

`Path`、`numpy`、`torch`、`LINUX`、`LOGGER`、`cpu_threads`、`check_requirements`、`BaseBackend`。

### 类 `NCNNBackend`（第 16–70 行）

#### `load_model`（第 23–53 行）

1. 日志；`check_requirements("ncnn", cmds="--no-deps")`；`import ncnn` → `self.pyncnn`、`self.net = Net()`。
2. **Vulkan 分支**：`device` 为 str 且 `startswith("vulkan")` → `use_vulkan_compute=True`，`set_vulkan_device(int(id))`，`self.device=torch.device("cpu")`。
3. **否则**：`use_vulkan_compute=False`。
4. **Linux**：`net.opt.num_threads = cpu_threads(原值)`。
5. **路径**：`w=Path(weight)`；非文件则 `next(w.glob("*.param"))`。
6. `load_param(w)`、`load_model(w.with_suffix(".bin"))`。
7. `apply_metadata(read_metadata(w))`。

#### `forward`（第 55–70 行）

**逐 batch 样本** `sample in im.cpu().numpy()`（仍为 NCHW numpy）：

1. `create_extractor()` 上下文。
2. `ex.input(net.input_names()[0], Mat(sample))`——首输入名绑定。
3. **输出**：对 `sorted(net.output_names())` 每个名 `ex.extract(x)[1]` 转 `np.array`（排序为 pnnx 输出顺序临时修复）。
4. 将该图所有输出列表 append 到 `outputs`。

**批维合并**：`zip(*outputs)` 对每个输出索引 `np.stack(y)`，得到 `list[np.ndarray]`，每元素 shape 首维为 batch。

无 UINT8 转换、无坐标反归一化；解码由上层完成。