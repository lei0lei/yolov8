# `ultralytics/utils/export/tensorflow.py`

TensorFlow 生态导出：**ONNX→SavedModel**（`onnx2saved_model`/`onnx2tf`）、**Keras→冻结 GraphDef `.pb`**（`keras2pb`）、**TFLite→Edge TPU**（`tflite2edgetpu`）。`tf_wrapper` 为 Detect/Pose 头打 TF 友好解码补丁。

## 逐段说明

### `tf_wrapper`（第 19–36 行）

- 对每个 `Detect`：绑定 `_tf_decode_boxes`；`Pose` 另绑 `_tf_kpts_decode`（Pose26 分支）。

### `_tf_decode_boxes`（第 39–50 行）

- 动态或 shape 变化时重建 anchors/strides。
- 用 grid 大小与 stride 归一化后 `decode_bboxes`+DFL；支持 angle。

### `_tf_kpts_decode`（第 53–65 行）

- reshape 关键点；grid 归一化；Pose26 与 2x+anchors-0.5 两式；3 维时 cat sigmoid。

### `onnx2saved_model`（第 68–204 行）

**依赖与环境（97–124）**

- 按 Python 版本 pin tensorflow；`onnx2tf` 无 deps 安装；GPU 时 onnxruntime-gpu；非 cuda 隐藏 TF GPU。

**INT8 准备（136–148）**

- 下载 onnx2tf 校准样例；有校准图则写临时 npy 与 `custom_input_op_name_np_data_path`。

**onnx.helper 补丁（150–162）**

- 补回 `float32_to_bfloat16`（ONNX 1.17 移除）。

**TopK 补丁（164–181）**

- 修补 onnx2tf TopK 的 `k_tensor` squeeze；只读 site-packages 则 warning。

**convert（184–204）**

- `onnx2tf.convert`：INT8 时 `output_integer_quantized_tflite`、关 batchmatmul unfold 等。
- 清理临时校准文件；重命名/删除多余 tflite；返回 `keras_model`。

### `keras2pb`（第 207–233 行）

- `tf.function` 包装具体函数；`convert_variables_to_constants_v2` 冻结；`write_graph` 到 `.pb`。

### `tflite2edgetpu`（第 236–303 行）

- 仅 Linux x86_64；查找系统或 Ultralytics 捆绑 `edgetpu_compiler`（缺则 AUTOINSTALL 下载 tar）。
- argv 调用编译器：`--search_delegate` 等；`check=True`。
- 返回 `{stem}_edgetpu.tflite` 路径。

### `gd_outputs`（第 306–312 行）

- 从 GraphDef 节点推断输出张量名（非 NoOp、无入边节点加 `:0`）。