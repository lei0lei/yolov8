# `tests/test_exports.py`

本模块系统验证 `Exporter` 对各部署格式的导出与推理，以及 `quantize` 参数规范化、INT8 校准数据加载、ONNX 并发锁、CoreML/QNN/NXP/LiteRT 等平台的元数据与 I/O 契约。大量矩阵测试标记 `slow` 并受 `--export-env` 过滤；树莓派跳过语义分割。多数用例在共享 `isolated_model` 或 `TASK2MODEL` 上导出后以 `SOURCE` smoke predict。

## 逐段说明

### `skip_rpi_semantic`（第 43–46 行）

树莓派上对 `task==semantic` 调用 `pytest.skip`。

### `test_export_torchscript`（第 49–56 行）

参数化 `nms`；导出后 predict 32 与 64，断言 `predictor.imgsz` 仍为 32。

### `test_export_torchscript_missing_names`（第 59–69 行）

`names=None` 的 detect/cls yaml；导出后 `names` 长度等于 head `nc`，末类名为 `class{nc-1}`。

### `test_export_onnx`（第 72–76 行）

动态 ONNX + 两种 nms 推理。

### `test_export_onnx_nms_conf`（第 79–93 行）

`nms=True, conf=0.0` 时 ONNX Greater 初始izers 含 0.0 阈值。

### `test_export_onnx_int8`（第 96–103 行，`slow`）

`int8` 与 `quantize=8` 别名均产出 `*_int8.onnx` 并可推理。

### `test_quantize_canonicalization`（第 106–132 行）

各类字符串/整数量化值规范化；非法值 `ValueError`。

### `test_quantize_deprecation`（第 135–142 行）

`half`/`int8` 映射到 `quantize`；显式 `quantize` 优先。

### `test_qnn_quantize_requires_w8a16`（第 145–150 行）

QNN 仅接受 `w8a16`；`quantize=8` 在 `validate_args` 断言失败。

### `test_modelopt_quantize_onnx_requires_int8_dataset`（第 153–156 行）

无 data 时 INT8 ModelOpt 早失败。

### `test_int8_calibration_validates_split`（第 159–166 行）

不存在 split 名抛 `FileNotFoundError`。

### `test_int8_calibration_fraction`（第 169–184 行）

标量与列表 fraction 决定校准集样本数（含临时分类目录）。

### `test_torch2onnx_serializes_concurrent_exports`（第 187–218 行）

四线程 fake export，`max_active==1`。

### `test_export_openvino` / `test_export_openvino_matrix`（第 221–252 行）

单测与全任务×dynamic×quantize×batch×nms 矩阵；清理目录。

### `test_export_onnx_matrix`（第 255–281 行，`slow`）

全组合；语义任务断言 mask dtype。

### `test_export_onnx_semantic_dnn`（第 284–290 行）

语义 ONNX + `dnn=True` 有 `semantic_mask`。

### `test_export_torchscript_matrix`（第 292–309 行，`slow`）

各任务 isolated 导出矩阵。

### `test_export_coreml_matrix` / `test_export_coreml`（第 312–381 行）

macOS 矩阵与 yolo11n 详细 metadata（IoU/conf、量化层、无 Error  stderr）。

### `test_export_coreml_rtdetr`（第 384–405 行）

RT-DETR 输出 shape 末维 300。

### `test_export_pb`（第 408–414 行）

默认 skip；Linux TensorFlow pb 导出推理。

### `test_export_paddle` / `test_export_mnn`（第 417–431 行）

Paddle 与 MNN smoke（MNN 须在 NCNN 前）。

### `test_export_mnn_rejects_unsupported_nms`（第 434–445 行）

非法 nms/batch/dynamic 组合 `ValueError`。

### `test_export_mnn_options` / `test_export_mnn_matrix`（第 448–478 行）

合法动态/nms 与全任务量化矩阵。

### `test_export_ncnn` / `test_export_ncnn_matrix`（第 481–496 行）

TORCH≥2.0；全任务 fp16 量化导出。

### `test_export_imx`（第 499–522 行）

Linux Py3.10–3.12；NMS 节点 `score_threshold` 随 `conf`。

### `test_export_rknn`（第 525–532 行）

x86 Linux；目录内含 `.rknn`。

### `test_export_executorch` / `test_export_executorch_matrix`（第 536–613 行）

`.pte` 与 `metadata.yaml` 存在；不跑设备推理。

### `test_export_nxp`（第 552–568 行）

Flatbuffer 含 `NeutronGraph`；metadata task 匹配。

### `test_export_litert_matrix`（第 571–594 行）

`.tflite` 且 I/O 均为 float32。

### `test_export_qnn`（第 616–643 行）

QNN EP 可用时导出 `_qnn.onnx`，无 `io_name_overrides` EPContext 属性。

### `test_export_stedgeai`（第 646–649 行）

INT8 whole-graph ONNX 可推理。

### `test_export_env_has_smoke`（第 652–655 行）

每个 EXPORT_ENV（除 base 特判）有 smoke 命令。

### `test_every_format_env_is_registered`（第 658–661 行）

`export_formats()` 中 Env 均在 `EXPORT_ENVS`。