# `ultralytics/engine/exporter.py`

本模块是 Ultralytics YOLO 家族模型**导出引擎**的核心实现：将训练好的 PyTorch 权重转换为 TorchScript、ONNX、OpenVINO、TensorRT、CoreML、TensorFlow/LiteRT 及多种 NPU/边缘芯片专用格式。对外主入口是 `Exporter.__call__(model)`，通常由 `YOLO.export()` / `yolo export` CLI 间接调用；`export_formats()` 与 `EXPORT_ENVS` 供文档、参数校验与 CI 隔离环境使用。导出前会融合模型、按格式改写检测头导出标志、可选包裹 `NMSModel`/`ClassMapModel`/`QNNModel`，并写入 `metadata` 供 `AutoBackend` 推理加载。模块内还提供 `validate_args` 做格式与量化参数合法性检查，`try_export` 装饰各 `export_*` 方法统一计时与成功/失败日志。

## 模块文档与常量（第 1～522 行）

### 第 1～71 行：模块 docstring

说明支持的导出格式表、pip 依赖（`ultralytics[export]`）、Python/CLI 用法示例，以及各格式对应的 `yolo predict model=` 路径。不含可执行代码。

### 第 73～160 行：导入分组

| 分组 | 内容 | 用途 |
|------|------|------|
| 标准库 | `json`, `os`, `shutil`, `tempfile`, `time`, `deepcopy`, `datetime`, `partial`, `Path` | 路径、临时目录、元数据、计时 |
| 数值与深度学习 | `numpy`, `torch` | 校准张量、导出 trace |
| 配置与数据 | `get_cfg`, `TASK2*`, `build_dataloader`, `check_*_dataset` | 导出参数与 INT8 校准数据 |
| 模型结构 | `nn.modules` 各检测头、`nn.tasks` 模型类 | 导出前头结点配置、任务判断 |
| 平台与工具 | `LOGGER`, `SETTINGS`, 平台常量、`callbacks`, `check_requirements`, `export/*` 子包 | 日志、依赖检查、各格式实际转换 |

### 第 163～321 行：`export_formats()`（163～321）

- 构建二维表 `x`：每行 `[Format, Argument, Suffix, CPU, GPU, Arguments, Env]`，共 24 种非 PyTorch 格式加首行 PyTorch。
- `return dict(zip(..., zip(*x)))`：转为列名字典，供 CLI 帮助、`validate_args`、以及 `Exporter.__call__` 中 `fmt_keys = fmts_dict["Arguments"][fmt]` 查询某格式允许的参数名列表。
- **结果**：调用方通过 `export_formats()["Argument"]` 得到 `torchscript`、`onnx` 等 CLI 关键字；`"Env"` 列对应 `EXPORT_ENVS` 的隔离测试环境 id。

### 第 324～484 行：`EXPORT_ENVS`

每个环境 id（`base`, `tensorflow`, `coreml`, …）描述：推荐 Python 版本、`extras`、额外 `requirements`、`indexes`、环境变量 `env`、CI `smoke` 命令。**不随导出自动安装**；供 `tests/test_exports.py` 与 `create-export-env.py` 使用。

### 第 487～522 行：量化格式集合

- `FP16_FORMATS` / `INT8_FORMATS` / `W8A16_FORMATS` / `W8A32_FORMATS` / `FP32_UNSUPPORTED_FORMATS`：声明哪些 `format=` 支持 `quantize=` 各取值。
- `QUANTIZE_PRECISIONS`：错误提示用的人类可读标签与格式集合配对。

## 逐段说明

### 第 525～560 行：`validate_args(format, passed_args, valid_args)`

1. 从 `export_formats()` 汇总全局导出参数列表 `export_args`，排除由 `Exporter` 单独处理的 `conf/iou/name/nms/quantize`。
2. `assert valid_args is not None`：该格式必须在表里列出 `Arguments` 列。
3. 与 `DEFAULT_CFG` 合并默认比较：若用户改了 `quantize`，按 16/8/w8a16/w8a32/32 分支 `assert` 落在对应 `*_FORMATS` 中；非 `FP32_UNSUPPORTED` 格式允许 FP32。
4. 循环 `export_args`：若某参数非默认值且不在该格式的 `valid_args` 里则报错。  
**目的**：在 `Exporter.__call__` 早期失败，避免 trace 一半才发现参数不支持。

### 第 563～583 行：`try_export` 装饰器

- 用 `get_default_args` 取被装饰函数签名默认值（主要是 `prefix` 日志前缀）。
- `outer_func`：`Profile()` 计时 → 调用 `inner_func` → 断言输出路径体积 > 0.1MB → 打成功日志；异常则打失败日志并 `raise`。  
**结果**：各 `export_*` 方法返回值（路径或元组）原样返回，调用方只关心路径字符串。

### 第 586～646 行：`Exporter` 类概述

持有 `args`、`callbacks`、导出期 `im/model/file/metadata/device` 等；列出全部 `export_*` 与校准辅助方法。实例化不加载模型，真正工作在 `__call__`。

### 第 648～660 行：`Exporter.__init__`

- `self.args = get_cfg(cfg, overrides)`。
- `iou` 为 `None` 时设为 `PREDICT_IOU`（嵌入 NMS 与 predict 一致）。
- 注册默认 callbacks 与 integration callbacks。

### 第 662～1130 行：`Exporter.__call__(model)` — 总览

**返回**：最终 artifact 路径字符串 `f`。  
**总流程**：规范化 `format` → 校验参数与平台 → 深拷贝并准备模型 → dry run → 元数据 → 按格式分支调用 `export_*` 或 TensorFlow 链 → 打印完成信息与 predict/val 提示。

#### 第 674～701 行：格式别名与纠错

- 小写 `format`；`tensorrt/trt`→`engine`；CoreML 多别名→`coreml`（保留 `mlmodel` 子模式）；`huawei/cann/om`→`ascend`；`vitis/...`→`xilinx`；废弃 `tflite/tfjs`→警告并改 `litert`。
- 不在表内：对 `pt` 提示已是 PyTorch；否则 `difflib` 60% 相似度自动纠正或 `ValueError`。

#### 第 702～721 行：设备选择

- TensorRT 无 device 时警告并 `device=0`；`dla:N` 拆出 `self.dla` 并强制 GPU 0。
- IMX 有 CUDA 时建议 GPU 0。
- `select_device`：无 device 或 `coreai/executorch/litert/nxp/openvino` 强制 CPU，其余用用户 device。

#### 第 723～974 行：参数与格式专项校验（节选逻辑）

| 分支 | 行为 |
|------|------|
| `validate_args` + 不支持嵌入 NMS | 警告并 `nms=None` |
| deepx/axelera/imx/... | 强制 INT8 或 qnn 的 w8a16 |
| hailo/imx/rknn/ascend/qnn/xilinx | 默认/校验 `name`（芯片、SoC、HTP） |
| WorldModel | 警告限制格式，deepcopy 时去掉 clip 权重 |
| `end2end` | 按格式禁用；LiteRT INT8 禁用 |
| QAT | 仅 onnx/engine，quantize 固定 8 |
| INT8 无 data | 填默认 `TASK2DATA` |
| CoreML/MNN/engine/edgetpu/... | 各文档所述断言与警告 |

**模型准备（975～1050）**：零张量 `im`；`requires_grad=False`；`eval().float().fuse()`；按格式套 `FXModel`/`tf_wrapper`/`stedgeai_wrapper`/`executorch_wrapper`；遍历模块设 `SPPF/Detect/...` 的 `export/dynamic/format/max_det`；semantic+qnn/coreml/ascend 包 `ClassMapModel`。

**Dry run（1051～1053）**：`nms` 且非 coreml/imx 用 `NMSModel`；否则 `model(im)`；FP16 时 GPU 上 `half()`。

**元数据与导出分发（1055～1130）**：

```
is_tf_format = fmt in {saved_model, pb, edgetpu}
若 is_tf_format:
    f, keras = export_saved_model()
    pb → export_pb(keras)
    edgetpu → export_edgetpu(tflite 路径)
否则:
    f = getattr(self, f"export_{fmt}")()
```

各格式**主路径**见下表（`@try_export` 内实现）：

| format 参数 | 方法 | 核心路径 |
|-------------|------|----------|
| torchscript | `export_torchscript` | `torch2torchscript`；可选 `NMSModel` |
| onnx | `export_onnx` | `torch2onnx` → 可选 onnxslim、ascend NMS 补丁、metadata、FP16/INT8 |
| openvino | `export_openvino` | `torch2openvino` + nncf INT8；目录 `*_openvino_model/` |
| engine | `export_engine` | **先** `export_onnx` → `onnx2engine`（GPU 必需） |
| coreml | `export_coreml` | `IOSDetectModel` 或 `NMSModel` → `torch2coreml`；detect+nms 再 `pipeline_coreml` |
| coreai | `export_coreai` | `torch2coreai` → `.aimodel` |
| saved_model | `export_saved_model` | **先** `export_onnx` → `onnx2saved_model`；写 metadata、tflite 元数据 |
| pb | `export_pb` | 依赖上一步 keras → `keras2pb` |
| edgetpu | `export_edgetpu` | SavedModel 链产出 tflite → `tflite2edgetpu` |
| litert | `export_litert` | `torch2litert`（8/w8a16/w8a32） |
| nxp | `export_nxp` | **先** `export_litert` → `tflite2nxp` |
| paddle | `export_paddle` | `torch2paddle` |
| mnn | `export_mnn` | **先** `export_onnx` → `onnx2mnn` |
| ncnn | `export_ncnn` | `torch2ncnn` |
| imx | `export_imx` | `torch2imx`（需 prior `FXModel`） |
| rknn | `export_rknn` | batch=1 **`export_onnx`** → 可选 dataset.txt → `onnx2rknn` |
| executorch | `export_executorch` | `torch2executorch` |
| axelera | `export_axelera` | `torch2axelera` + 校准 |
| deepx | `export_deepx` | **`export_onnx`** → `onnx2deepx` |
| qnn | `export_qnn` | 临时 `QNNModel`+NHWC `im` → **`export_onnx`** → `onnx2qnn` |
| hailo | `export_hailo` | 定制 `end_nodes`、`export_onnx` → Hailo `ClientRunner` → `.hef` |
| ascend | `export_ascend` | **`export_onnx`**（opset≤17）→ `onnx2ascend` |
| xilinx | `export_xilinx` | **`export_onnx`** → `onnx2xilinx` |
| stedgeai | `export_stedgeai` | 改输出名 → **`export_onnx`**；INT8 时整图 `onnx_int8_quantize` |

### 第 1132～1178 行：`get_int8_calibration_dataloader`

- 深拷贝 `args`，分类任务 `imgsz` 用精确 (h,w)，检测用 `max(imgsz)`；`rect=False`；`split` 默认 val；NDJSON 转 YOLO 格式。
- **分类**：`ClassificationDataset`，transform 只保留 resize/crop + `PILToTensor` 输出 uint8。
- **检测等**：`build_yolo_dataset` val 模式；同步 `LetterBox.new_shape`。
- 图像数 < batch：用 `InfiniteDataLoader` 重复索引凑满 batch；否则 `build_dataloader` `drop_last=True`。  
**结果**：供 OpenVINO nncf、ONNX INT8、QNN、TensorRT INT8 等读取。

### 第 1180～1188 行：`_int8_calibration_images`

- 遍历 loader，将 batch 插值到 `imgsz` 后堆成 BHWC float32 numpy。Hailo 等需要固定 numpy 校准集的路径用此。

### 第 1190～1201 行：`export_torchscript`

- 模型：`NMSModel` 或原模型；`torch2torchscript(im, metadata)` → `.torchscript`。

### 第 1203～1319 行：`export_onnx`（分段）

**依赖与 opset（1204～1228）**：按 rknn/stedgeai 等选 onnx 版本范围；simplify 或 INT8 时拉 onnxruntime；`best_onnx_opset`；按 task/RT-DETR/nms 设 `min_opset`。

**动态维与 NMS（1232～1248）**：segment 双输出名；dynamic 字典按 Detection/Segmentation/Depth 设 batch/anchors 或 H/W；OBB+nms 强制 simplify。

**trace（1250～1260）**：QAT 时 model/im 先 `.cpu()`；`arange_patch` 上下文内 `torch2onnx`。

**后处理（1262～1318）**：可选 onnxslim；ascend 给 NMS 加 score threshold 输入；写 metadata_props；IR≤10；CPU FP16 用 onnxruntime transformers；INT8 非 QAT 调 `onnx_int8_quantize` 并删 FP32 中间文件。

### 第 1321～1370 行：`export_openvino`

- macOS 15.4+ 要求较新 openvino；`TORCH_2_1`。
- 内部 `serialize`：写 rt_info、`compress_to_fp16`、存 `metadata.yaml`。
- INT8：`nncf.Dataset(loader, _transform_fn)`。
- `torch2openvino` → 目录 `*_openvino_model/*.xml`。

### 第 1372～1383 行：`export_paddle`

- `torch2paddle` 到 `*_paddle_model/`。

### 第 1385～1407 行：`export_litert`

- 平台：Linux x86 或 macOS；`torch2litert` 传入 quantize 与可选校准 loader。

### 第 1409～1415 行：`export_nxp`

- Linux x86；**链式** `tflite2nxp(self.export_litert(), ...)`。

### 第 1417～1428 行：`export_mnn`

- **链式** `onnx2mnn(self.export_onnx(), ...)`。

### 第 1430～1443 行：`export_ncnn`

- `torch2ncnn` 输出 `*_ncnn_model/`。

### 第 1445～1519 行：`export_coreml`

- `mlmodel` 参数选 legacy `.mlmodel`。
- detect+nms：`IOSDetectModel`；segment/pose+nms：`NMSModel`。
- dynamic：`ct.RangeDim` batch 与 H/W；否则 `ImageType` RGB/GRAY。
- `torch2coreml`；detect pipeline 加 Apple NMS；分类写 preview metadata；保存 `.mlpackage` 或 `.mlmodel`。

### 第 1521～1544 行：`export_engine`

- 断言 GPU；**必须先** `export_onnx()`；`onnx2engine`（workspace、quantize、dynamic、dla、INT8 dataset）。

### 第 1546～1580 行：`export_saved_model`

- macOS Py3.13 禁止；删旧 `_saved_model` 目录；RT-DETR opset≤19；强制 simplify 后 `export_onnx`；`onnx2saved_model`；遍历 tflite 写 metadata zip。

### 第 1582～1587 行：`export_pb`

- `keras2pb` → `.pb`。

### 第 1589～1608 行：`export_axelera`

- Linux 非 ARM Docker；torch≥2.8；`torch2axelera` + 校准。

### 第 1610～1623 行：`export_executorch`

- torch≥2.9；`torch2executorch`；INT8 传校准 loader。

### 第 1625～1641 行：`export_coreai`

- macOS 26+ ARM 或 Linux x86；`torch2coreai` → `.aimodel`。

### 第 1643～1650 行：`export_edgetpu`

- `tflite2edgetpu` + `_add_tflite_metadata`。

### 第 1652～1691 行：`export_rknn`

- opset≤19；`im` 取 batch 1；INT8 写 symlink dataset.txt；`onnx2rknn` 指定 head 截断路径。

### 第 1693～1711 行：`export_ascend`

- `_check_atc`；opset≤17；**`export_onnx`** → `onnx2ascend`。

### 第 1713～1733 行：`export_imx`

- Linux，Python 3.10～3.12；`torch2imx` 传 conf/iou/max_det 与校准 partial。

### 第 1735～1749 行：`export_deepx`

- Linux x86；**`export_onnx`** → `onnx2deepx`。

### 第 1751～1771 行：`export_qnn`

- try/finally 恢复 model/im；`QNNModel` + NHWC；**`export_onnx`**；`onnx2qnn` 到 `*_qnn.onnx`。

### 第 1773～1788 行：`export_xilinx`

- Python 3.11～3.13；先校验 data；**`export_onnx`** → `onnx2xilinx`。

### 第 1790～1801 行：`export_stedgeai`

- 改文件名为 `*_stedgeai.onnx`；**`export_onnx`**；INT8 原地 whole_graph quantize。

### 第 1803～1943 行：`export_hailo`

- 需 `hailo_sdk_client`；按 task 选 ONNX `end_nodes`（分类 Softmax、semantic ArgMax、depth 最后一层 conv、detect/segment/pose/obb 各分支 Conv）；opset 11；`export_onnx`；segment 追加 proto 末节点；`ClientRunner.translate/optimize/compile` 写 `.hef` 与 `metadata.yaml`；finally 删中间 onnx。

### 第 1945～1957 行：`_add_tflite_metadata` / `_transform_fn`

- zip 追加 `metadata.json`；校准项 uint8→float/255，batch 维可选。

### 第 1959～1966 行：`add_callback` / `run_callbacks`

- 标准事件 `on_export_start` / `on_export_end`。

### 第 1969～1990 行：`ExportWrapper`

- 存 `_model`；`__getattr__` 转发到被包装模型，使 `model.model[-1]` 等仍可用。

### 第 1992～2006 行：`QNNModel.forward`

- 输入 NHWC；`permute` 成 NCHW 调 `_model`；ONNX trace 时转置可并入 QNN 图。

### 第 2009～2036 行：`ClassMapModel`

- semantic 专用：`forward` 对 logits `argmax` 或二值 `gt(0)`，dtype uint8（≤256 类）或 int32。

### 第 2039～2130 行：`NMSModel`

**`__init__`**：`obb` 标志；TensorFlow 导出 `is_tf` 影响 mask 分支。

**`forward(x)` — 第 1 段（2065～2077）**：`model(x)` → 转置为 (B, anchors, C)；算 `extra_shape`；dynamic batch 时 pad；ONNX+OBB dynamic 时 cat 零行防 topk 形状问题；split 为 boxes / 各类分数 / extras；`max` 得 class 分。

**第 2 段（2078～2102）**：按 batch 循环；`conf` 过滤；TF/ONNX OBB 用 topk 代替空 mask；CoreML 保 top1 anchor；OBB 用 `batch_probiou`+`TorchNMS.fast_nms`，否则 `torchvision.ops.nms`；类偏移实现 class-wise NMS。

**第 3 段（2117～2130）**：拼接 box/conf/cls/extra；pad 到 `max_det`（CoreML 用 1D pad）；segment 返回 `(out, proto)`，否则 `out[:bs]`。

---

**调用关系小结**：`ultralytics.engine.model.Model.export` → `Exporter(overrides)(self.model)`；测试与 CLI 同路径。推理侧 `AutoBackend` 根据后缀与环境 id 加载本模块产物。