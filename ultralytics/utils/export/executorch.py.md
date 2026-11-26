# `ultralytics/utils/export/executorch.py`

**ExecuTorch**：`torch.export` → **XnnpackPartitioner** 降级为 **`.pte`**；可选 XNNPACK 静态 INT8（Detect 头排除量化）。

## 逐段说明

### `executorch_wrapper`（第 16–24 行）

- 对 `Pose`/`Pose26` 绑定 `_executorch_kpts_decode`（与 TF 类似但 4D broadcast）。

### `_executorch_kpts_decode`（第 27–39 行）

- anchors/strides 扩维；Pose26 与 v8 公式分支；3 维关键点 cat sigmoid；reshape 输出。

### `torch2executorch`（第 42–102 行）

1. `check_executorch_requirements`。
2. **若** 有 `calibration_dataset`（需 executorch≥1.6）：
   - `XNNPACKQuantizer` 全局 per-channel 对称；`Detect` 模块类型设为不量化。
   - `prepare_pt2e` → 校准前向 `/255` → `convert_pt2e`。
3. `to_edge_transform_and_lower(export(...), XnnpackPartitioner).to_executorch()`。
4. 清空 output_dir，写 `model.pte`；可选 metadata.yaml。
5. 返回目录路径。