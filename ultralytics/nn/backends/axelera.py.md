# `ultralytics/nn/backends/axelera.py`

本后端面向 **Axelera Metis** 加速器，加载编译产物 **`.axm`**（在权重目录内 `rglob` 查找）。依赖 **`axelera-rt`**（版本与 `ultralytics.utils.export.axelera.AXELERA_SDK` 对齐），从 Axelera 私有 PyPI 索引安装；运行时使用 `axelera.runtime.op` 加载并 `optimized()` 优化图。输入为 **BCHW、[0,1]**；输出为 NumPy 单数组或按输出头分的列表，批处理经 `batch()` 再沿 batch 维拼接。

## 逐段说明

### 模块导入（第 1–12 行）

`Path`、`numpy`、`torch`、`check_requirements`、`BaseBackend`。

### 类 `AxeleraBackend`（第 15–68 行）

#### `load_model`（第 21–43 行）

1. 从导出工具模块读取 `AXELERA_SDK` 版本号。
2. **`check_requirements`**：要求 `axelera-rt=={AXELERA_SDK}`，并附加 Axelera artifactory 的 `--extra-index-url`；失败则 `ModuleNotFoundError`。
3. `from axelera.runtime import op`。
4. 在 `Path(weight)` 下 `rglob("*.axm")`，无文件则 `FileNotFoundError`。
5. `self.model = op.load(path).optimized()`。
6. `apply_metadata(read_metadata(found))`。

#### `forward`（第 45–68 行）

**输入**：`im` 先 `cpu()`，仍为 BCHW [0,1]。

**分支一：`im.shape[0] == 1`（单张）**

- 直接 `return self.model(im)`，返回该 API 的原生结构（单输出数组或多输出）。

**分支二：批大小 > 1**

1. `outputs = self.model.batch(im)`：整批作为一次调度输入；`batch()` 的变参表示**每个模型输入**各一个张量，而非每张图一个参数。
2. **多输出头**：若 `outputs[0]` 为 `list` 或 `tuple`（如检测 + mask），对 `zip(*outputs)` 的每个输出流做 `np.concatenate(..., axis=0)`，得到与 batch 对齐的列表。
3. **单输出**：否则 `np.concatenate(outputs, axis=0)` 合并各图结果。

每张图结果保留 leading 维为 1，故用 `concatenate` 而非 `stack`。