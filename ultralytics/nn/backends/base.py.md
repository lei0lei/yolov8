# `ultralytics/nn/backends/base.py`

本文件定义推理后端的抽象契约与跨格式公共能力。`BaseBackend` 规定 `load_model` 与 `forward(im)` 两大接口，输入约定为 BCHW、归一化到 [0,1] 的 `torch.Tensor`；输出为各格式原生张量或数组，由 `AutoBackend` 与 predictor 再做任务侧后处理。模块级 `_read_proto_map` 在不导入 ONNX/CoreML 运行时的情况下解析 protobuf 内嵌元数据。`read_metadata` 按后缀从 engine 头、zip 内 JSON、ONNX metadata_props、侧车 `metadata.yaml` 等路径抽取导出信息。`apply_metadata` 将 stride、names、end2end、imgsz 等写回实例属性；`denormalize` 在 host 上将检测/姿态等归一化坐标还原为像素。

## 逐段说明

### `_read_proto_map`（第 17–57 行）

**职责**：纯字节流解析嵌套 protobuf `map<string,string>`。

**参数**：`file` 为模型路径；`path` 为字段号元组，末级为 map 所在消息。

**实现要点**：`mmap` 映射文件；内部 `fields` 生成器按 varint 标签解析 length-delimited 字段；沿 `path` 下钻后在 map 条目中解码 key/value 为字符串字典。大权重 blob 用 memoryview 切片避免整段拷贝。

### `BaseBackend` 类（第 60–251 行）

#### `__init__`（第 84–105 行）

初始化 `device`、`fp16`、`nhwc=False`、默认 `stride=32`、空 `names`、`task=None`、`batch=1`、`channels=3`、`end2end/dynamic/base_model` 标志、`metadata` 与 `model=None`，并调用子类 `load_model(weight)`。

#### `load_model`（第 107–114 行，抽象）

子类必须实现：从路径或 `nn.Module` 加载运行时对象到 `self.model`。

#### `forward`（第 116–126 行，抽象）

子类必须实现：对 `im` 执行推理，返回格式依后端而定（可能需 predictor 解码）。

#### `__call__`（第 128–130 行）

转发至 `forward`。

#### `engine_header`（第 132–148 行，静态）

读取 TensorRT `.engine` 文件前 4 字节小端长度 + JSON 元数据头。返回 `(引擎字节偏移, dict)`；无合法头则 `(0, {})`。

#### `read_metadata`（第 150–202 行，静态）

按后缀分支：`.engine` 用 `engine_header`；`.tflite`/`.torchscript` 读 zip 内 `metadata.json` 或 `extra/config.txt`；`.onnx` 或 `*_imx_model` 目录用 `_read_proto_map`；`.mlpackage`/`.mlmodel` 读 userDefined；`.mnn` 解析 flatbuffer `bizCode`；`.aimodel` 读 `metadata.json`；其余尝试同级或父级 `metadata.yaml`（`.pb` 另搜 `*_saved_model*/metadata.yaml`）。任意异常返回 `{}`。

#### `apply_metadata`（第 204–232 行）

空则直接返回。保存 `self.metadata`；对 `stride`/`batch`/`channels` 转 int；对 `imgsz`、`names` 等字符串 `literal_eval`；`end2end` 合并 `args.nms`；`dynamic` 来自 `args.dynamic`；最后 `setattr` 全部键到实例（含 `kpt_shape` 等）。

#### `denormalize`（第 234–251 行）

**输入**：`x` 为 `(B, 4+nc+extra, N)` 预测或 end2end 时 `(B, N, 6+extra)`；`w,h` 为输入宽高；`end2end` 控制是否先转置为 channels-first 再写回。

**操作**：x/y 坐标乘 `w`/`h`；`task=="pose"` 时对关键点列按 `kpt_shape` 与布局索引做同样缩放。

## 与统一接口的对齐

- **加载**：子类在 `load_model` 末尾通常调用 `apply_metadata(read_metadata(...))`，使 `names`、`task`、`stride`、`end2end` 与 PyTorch  checkpoint 一致。
- **前向**：统一入口为 BCHW float 输入；部分后端（CoreML、TFLite）由 `AutoBackend` 在调用前做 NHWC 或 dtype 转换。
- **输出**：`forward` 返回原生结构；检测头解码、NMS、mask proto 组装仍在 `engine/predictor` 与 `AutoBackend` 层完成，后端不负责完整 Results 对象。