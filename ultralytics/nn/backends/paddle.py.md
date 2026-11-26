# `ultralytics/nn/backends/paddle.py`

本后端加载百度 **PaddlePaddle 推理** 格式：**`model.json`（结构）+ `*.pdiparams`（权重）**，支持目录或单独的 `.pdiparams`（同目录找 `model.json`）。依赖 `check_paddle()` 后的 **`paddle.inference`**。CUDA 可用且 `device` 非 CPU 时 `enable_use_gpu(2048MB, device_id)`。输入 **BCHW float32** 经 `copy_from_cpu`；各命名输出 `copy_to_cpu` 组成列表。

## 逐段说明

### 模块导入（第 1–13 行）

`Path`、`numpy`、`torch`、`LOGGER`、`check_paddle`、`BaseBackend`。

### 类 `PaddleBackend`（第 16–69 行）

#### `load_model`（第 23–56 行）

1. **`cuda`**：`device` 为 torch.device、`cuda.is_available()`、`device.type != "cpu"`。
2. 日志；`check_paddle()`；`import paddle.inference as pdi`。
3. **`w = Path(weight)` 解析文件对**：
   - **目录**：`rglob("*.json")` 与 `rglob("*.pdiparams")` 各取第一个。
   - **`.pdiparams` 文件**：`model_file = with_name("model.json")`，`params_file = w`。
   - 任缺或文件不存在 → `FileNotFoundError`。
4. `Config(model, params)`；**cuda 真**则 `enable_use_gpu(...)`。
5. `create_predictor`；缓存首输入 `input_handle` 与全部 `output_names`。
6. `apply_metadata(read_metadata(w))`。

#### `forward`（第 58–69 行）

1. `input_handle.copy_from_cpu(im.cpu().numpy().astype(np.float32, copy=False))`——无 NHWC 切换。
2. `predictor.run()`。
3. `return [get_output_handle(name).copy_to_cpu() for name in output_names]`——顺序与 Paddle 图输出名一致，不在此合并或解码。