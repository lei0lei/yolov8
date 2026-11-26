# `ultralytics/nn/backends/rknn.py`

本后端仅在 **Rockchip** 设备（`is_rockchip()`）上加载 **`.rknn`**（文件或目录内 `rglob`），依赖 **`rknn-toolkit-lite2`** 的 `RKNNLite`：`load_rknn` + `init_runtime`，返回码非 0 抛 `RuntimeError`。输入文档写 BHWC [0,1]，实际前向为 **`uint8`**（×255）。若同目录存在 **`tail.onnx`**，则在 CPU 上用 **ONNX Runtime** 对 NPU 输出的每 batch 切片做浮点尾部解码，再沿 batch `concatenate`。元数据从 `.rknn` 旁路读取。

## 逐段说明

### 模块导入（第 1–13 行）

`Path`、`numpy`、`torch`、`LOGGER`、`ONNXRUNTIME_BUILDS`、`check_requirements`、`is_rockchip`、`BaseBackend`。

### 类 `RKNNBackend`（第 16–78 行）

#### `load_model`（第 23–59 行）

1. **`not is_rockchip()`** → `OSError`。
2. 日志；`check_requirements("rknn-toolkit-lite2")`；`RKNNLite()`。
3. **路径**：非文件则 `next(w.rglob("*.rknn"))`。
4. **`load_rknn`**：返回值 `ret != 0` → `RuntimeError`。
5. **`init_runtime`**：同上检查 `ret`。
6. **`self.tail = None`**；若 `w.parent / "tail.onnx"` 存在：
   - `check_requirements([ONNXRUNTIME_BUILDS])`；
   - `InferenceSession(tail.onnx, providers=["CPUExecutionProvider"])`。
7. `apply_metadata(read_metadata(w))`。

#### `forward`（第 61–78 行）

1. **预处理**：`im.cpu().numpy() * 255` → `astype("uint8")`（与 RKNN 量化输入契约一致；注意与类 docstring 的 float 表述相比此处为 0–255 整型）。
2. **`y = model.inference(inputs=[im])`**——列表包一层输入；`y` 为 NPU 原始多输出列表（每输出 numpy，含 batch 维）。
3. **分支 `self.tail` 存在**（INT8 导出，头在主机浮点解码）：
   - 对 batch 索引 `i`：`feeds = {tail输入名: v[i:i+1].astype(float32) for 每输入 v in y}`。
   - 每张图 `tail.run(None, f)`；`zip(*runs)` 后每输出流 `np.concatenate`。
4. **无 tail**：直接 `return y`。

不在此做 NCHW/NHWC 显式转换；布局由导出与 RKNN 图约定。输出拆分：无 tail 时保持 RKNN 多输出列表；有 tail 时 list 元素为拼接后的 numpy 张量。