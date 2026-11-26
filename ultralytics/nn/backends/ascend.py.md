# `ultralytics/nn/backends/ascend.py`

本后端面向**华为昇腾（Ascend）NPU**，加载经 CANN 编译的离线模型文件 **`.om`**（通常在导出目录内递归查找）。运行时依赖华为 CANN 与 **`ais_bench`** 包中的 `InferSession`（封装 pyACL），在指定 NPU 设备上执行推理。输入约定为 Ultralytics 统一的 **BCHW、float、[0,1]** 张量；输出为 NumPy 数组或数组列表，由上层 predictor 按元数据解析。

## 逐段说明

### 模块导入（第 1–12 行）

引入 `Path`、`numpy`、`torch`、`LOGGER` 以及基类 `BaseBackend`。本文件无模块级函数。

### 类 `AscendBackend`（第 15–66 行）

继承 `BaseBackend`，实现昇腾 `.om` 模型的加载、资源释放与前向推理。

#### `load_model`（第 22–49 行）

**加载流程：**

1. **依赖检查**：`try` 从 `ais_bench.infer.interface` 导入 `InferSession`；`except ImportError` 时抛出带文档链接的 `ImportError`，说明需安装 CANN 与 `ais_bench`。
2. **日志**：记录正在加载的路径。
3. **定位 `.om`**：`Path(weight)` 后对目录 `rglob("*.om")` 取第一个；若 `found is None` 则 `FileNotFoundError`。
4. **创建会话**：`InferSession(device_id, om_path)`，其中 `device_id` 为 `self.device.index`，无 index 时用 `0`。
5. **元数据**：`apply_metadata(read_metadata(found))`，从 `.om` 邻近导出元数据恢复 stride、task 等。

不在此做图像预处理；仅建立 NPU 推理会话。

#### `__del__`（第 51–54 行）

析构时若 `self.model` 存在，调用 `model.free_resource()` 释放昇腾设备侧资源。

#### `forward`（第 56–66 行）

**前向：**

1. 输入 `im`：BCHW、[0,1] 的 `torch.Tensor`。
2. **预处理**：`im.cpu().numpy()`，保持 NCHW float，无额外归一化或布局变换。
3. **推理**：`self.model.infer([numpy_batch])`，传入单元素列表（一批一张图或整批作为 SDK 接受的输入）。
4. **输出拆分**：`y = infer 返回值`；若 `len(y) == 1` 返回 `y[0]`（单个 `np.ndarray`），否则返回整个列表 `y`（多输出头）。不在此解码检测框，解码由共享后处理完成。