# `ultralytics/nn/backends/tensorflow.py`

本文件封装 TensorFlow SavedModel、冻结 GraphDef（`.pb`）与 Edge TPU `.tflite` 三种路径。SavedModel 调用 `serving_default` 签名；pb 经 `wrap_frozen_graph` 裁到导出节点；TFLite 经 delegate 上 Edge TPU，处理 int8 量化输入输出并在 host 上对 3D 检测输出 `denormalize`。`forward` 假定输入为 BHWC numpy（`AutoBackend` 转换），返回 `list[np.ndarray]`，分割任务修正 proto 维序与 NHWC→NCHW。

## 逐段说明

### `TensorFlowBackend` 类（第 16–149 行）

#### `__init__`（第 23–34 行）

`format` ∈ `{saved_model, pb, edgetpu}` 后 `super().__init__`。

#### `load_model`（第 36–90 行）

- **saved_model**：`tf.saved_model.load`；`read_metadata` 目录侧车。
- **pb**：读 GraphDef；`wrap_frozen_graph` 输入 `x:0`、输出 `gd_outputs(gd)`；metadata 同 saved_model 兄弟目录。
- **edgetpu**：`tflite_runtime` 或完整 TF；`tpu:N` 解析设备；`load_delegate` 平台库；`allocate_tensors`；记 `input_details`/`output_details`；`device` 置 CPU（逻辑设备）。

#### `forward`（第 92–149 行）

`im` 为 BHWC numpy。

- **saved_model**：`serving_default(im)`，非 list 则包成 list。
- **pb**：`frozen_func(x=constant(im))`。
- **edgetpu**：整数量化输入 scale/zero_point 反量化前向；每输出反量化；3D 且非 semantic argmax 时 `denormalize`（end2end 由末维 6 或 `self.end2end` 判断）。

**任务后处理（第 140–148 行）**：

- **segment**：若 proto 非 4D 则 reverse 顺序；end2end 仅留 `(B,N,6)`；否则 proto `transpose` 为 `(B, nm, H, W)`。
- **semantic/depth**：单输出 4D 时 `(0,3,1,2)` NHWC→NCHW。

末行将 TF tensor 转 `np.ndarray`。

## 与统一接口的对齐

- **加载**：`metadata.yaml` 与 TFLite zip metadata；`task` 驱动 forward 分支。
- **前向**：BHWC；与 TF 导出一致。
- **输出**：检测 `(B,N,C)` 像素或归一化坐标（edgetpu 已 denormalize）；分割 `(det, proto)` 维序与 PyTorch predictor 一致。