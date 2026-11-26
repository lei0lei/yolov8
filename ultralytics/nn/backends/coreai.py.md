# `ultralytics/nn/backends/coreai.py`

本后端在 **macOS 26+、Apple Silicon** 上运行 **Core AI** 导出的 **`.aimodel`** 资源（目录路径传入 `AIModel.load`）。依赖 **`coreai-torch>=0.4.3`** 与 `coreai.runtime`（`AIModel`、`NDArray`）。Core AI Python API 为异步，后端自建事件循环并用 `run_until_complete` 驱动加载与推理。输入 **BCHW [0,1]**；输出为 **float `torch.Tensor` 列表**，顺序与模型声明的输出名一致。

## 逐段说明

### 模块导入（第 1–13 行）

含 `asyncio`、`Path`、`torch`、`LOGGER`、`check_requirements`、`BaseBackend`。

### 类 `CoreAIBackend`（第 16–57 行）

#### `load_model`（第 23–41 行）

1. 日志记录加载路径。
2. `check_requirements("coreai-torch>=0.4.3")`。
3. `asyncio.new_event_loop()` 存为 `self._loop`。
4. **`AIModel.load(w)`**：经 `_await` 包装后在循环中 `run_until_complete` 得到 `model`。
5. **`load_function("main")`**：异步加载主计算图，得到 `self._function`。
6. 从 `descriptor` 读取：首输入名 `self._input_name`、全部输出名 `self._output_names`。
7. **`fp16` 推断**：若首输入 dtype 字符串含 `"float16"` 则 `self.fp16 = True`。
8. `apply_metadata(read_metadata(w))`（目录级元数据）。

#### `forward`（第 43–57 行）

1. 构造输入字典：`{input_name: NDArray(im.cpu().numpy())}`，NCHW float numpy。
2. `self._function(inputs)` 异步执行，经 `_await` 与 `run_until_complete` 得 `out`。
3. **输出取值**：若 `out` 为 `dict`，按 `self._output_names` 顺序取；否则 `list(out)`。
4. **转为 torch**：对每个值，若有 `.numpy()` 则取之，否则直接用；`torch.from_numpy(...).float()` 组成列表返回。不在此做任务专用解码。

### 函数 `_await`（第 60–64 行）

模块级异步辅助：`inspect.isawaitable(value)` 为真则 `await value`，否则原样返回。用于兼容同步/异步混用的 runtime API。