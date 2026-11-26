# `ultralytics/nn/backends/deepx.py`

本后端面向 **DEEPX NPU**，加载 **`.dxnn`** 编译模型（目录内 `rglob`）。依赖 DEEPX **DX-Runtime** 的 Python 包 **`dx_engine.InferenceEngine`**。与 Ultralytics 默认输入不同：前向时将每张图从 **BCHW float [0,1]** 转为 **HWC uint8 [0,255]** 再送入引擎。输出按输出头收集后沿 batch 维 `concatenate`，单头返回数组，多头返回列表。

## 逐段说明

### 模块导入（第 1–12 行）

`Path`、`numpy`、`torch`、`LOGGER`、`BaseBackend`。

### 类 `DeepXBackend`（第 15–70 行）

#### `load_model`（第 21–48 行）

1. **`try`/`except`** 导入 `InferenceEngine`；失败则 `ImportError` 并指向 DeepX 文档。
2. 日志、路径 `w = Path(weight)`，`rglob("*.dxnn")`，无文件 `FileNotFoundError`。
3. `self.model = InferenceEngine(str(found))`。
4. `apply_metadata(read_metadata(found))`。

#### `forward`（第 50–70 行）

**逐样本循环**（`for sample in im.cpu().numpy()`，即 batch 维拆开）：

1. **预处理单张**：`transpose (1,2,0)` 得 HWC；乘 255、`clip` 到 [0,255]、`astype(uint8)`、`ascontiguousarray`。
2. **推理**：`self.model.run([sample])`，对每个返回值 `map(np.asarray, ...)` 枚举。
3. **按输出索引累积**：`i` 为输出头索引；`outputs[i]` 为列表，追加 `out`（若 `out.ndim` 且 `shape[0]==1` 用原数组，否则 `out[None]` 保 batch 维）。
4. **批合并**：`y = [np.concatenate(x, axis=0) for x in outputs]`。
5. **返回**：`len(y)==1` 则 `y[0]`，否则 `y`。

多 batch 在 Python 层串行逐张推理后拼接，无 GPU 张量路径。