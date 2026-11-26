# `ultralytics/utils/benchmarks.py`

本模块提供两类能力：`benchmark()` 对 YOLO 模型按 `export_formats()` 列表（或单一 format）依次导出、预测烟雾测试与验证，汇总 mAP（或任务指标）、推理时延、FPS 与文件大小到 Polars DataFrame；`ProfileModels` 则针对给定权重/YAML/ONNX 路径做 ONNX Runtime 与 TensorRT 纯速度剖析并打印对比表。内置大量平台、Python 版本与 YOLOWorld 相关的 assert 跳过规则，避免 CI 死锁或 segfault。

## 逐段说明

### `benchmark`（第 74～295 行）— 初始化与数据准备

- **第 112～117 行**：`_handle_deprecation` 处理 device/quantize；`check_imgsz`；断言方形 imgsz。
- **第 120～127 行**：懒 import polars 并配置表格显示（全列、ASCII 边框等）。
- **第 129～132 行**：Intel 格式时保留 device 字符串否则 `select_device`。
- **第 133～134 行**：str/Path 则 `YOLO(model)`。
- **第 135～136 行**：默认数据集与指标键来自 `TASK2DATA`/`TASK2METRIC`。
- **第 137～143 行**：YOLOE/YOLOWorld 无 lrpc 时可能转换 data、deepcopy 并 `set_classes` 对齐验证词汇表。

### `benchmark` — 格式循环（导出前检查）

- **第 148～157 行**：指定 `format` 时校验在 export_formats 参数集合内；Py3.13 全量 benchmark 可跳过部分 TF 格式。
- **第 151～219 行**：逐格式 assert：edgetpu/coreml/paddle/mnn/ncnn/imx/rknn/executorch/axelera/xilinx/litert 等环境限制；CPU/GPU 与格式 cpu/gpu 标志匹配。
- **第 154～155 行**：`format_arg` 不匹配则 continue。

### `benchmark` — 导出、预测、验证

- **第 222～224 行**：原生 PyTorch 行深拷贝模型；否则 `model.export(...)` 得路径，RTDETR 用 RTDETR 加载否则 YOLO。
- **第 237～238 行**：后缀校验 export 成功；emoji 标记导出成功。
- **第 241～252 行**：按格式 assert 推理环境；`predict` bus.jpg 烟雾测试。
- **第 255～266 行**：`val` batch=1、`conf=0.001` 等固定设置取 metric 与 inference speed。
- **第 268～269 行**：FPS = 1000/(speed+eps)。
- **第 270～274 行**：异常时 verbose 可将非 AssertionError 升级；否则 error 日志；失败行填 None 指标。

### `benchmark` — 汇总输出

- **第 277～288 行**：`check_yolo` 打环境；构 Polars DF；日志与追加 `benchmarks.log`。
- **第 290～293 行**：verbose 为 float 时 assert 所有非 NaN 指标 &gt; floor。
- **第 295 行**：返回 `df_display`。

### 类 `ProfileModels`（第 298～661 行）

#### `__init__`（第 332～365 行）

- 保存 paths、计次参数、min_time、imgsz、`quantize`、`trt`；`select_device`。

#### `run`（第 367～419 行）

- `get_files()` 为空则 warning 返回 []。
- 对每个 file：pt/yaml 则 fuse、info、可选 export engine、export onnx；onnx 则 placeholder info。
- `profile_tensorrt_model`、`profile_onnx_model`；生成行与 dict；`print_table`；返回 output 列表。

#### `get_files`（第 421～439 行）

- 目录则按 stem 去重收集 onnx/yaml/pt；单文件按后缀或 glob；打日志 sorted paths。

#### `get_onnx_model_info`（第 441～451 行）

- 静态返回四个 0.0（ONNX 不解析 FLOPs 等）。

#### `iterative_sigma_clipping`（第 453～473 行）

- 循环最多 max_iters：均值±sigma*std 保留样本；不变则 break。

#### `profile_tensorrt_model`（第 475～510 行）

- 无文件或未启用 trt 返回 (0,0)。
- YOLO engine、uint8 零图；3 轮 warmup×num_warmup_runs；num_runs = max(min_time/elapsed 推算, num_timed_runs*50)。
- 每次 predict 取 speed inference ms；sigma clip 后 mean/std。

#### `check_dynamic`（第 512～515 行）

- shape 任一维非 int 或负则动态。

#### `profile_onnx_model`（第 517～584 行）

- ORT SessionOptions 全图优化、8 线程；ONNXBackend CPU fp32。
- 为每个输入按类型建随机张量，动态维用 default 填充。
- warmup 与 num_runs 类似（系数为 num_timed_runs 非 50x）。
- perf_counter 计 forward 毫秒；clip max_iters=5。

#### `generate_table_row` / `generate_results_dict`（第 586～635 行）

- 格式化 Markdown 行与 W&B 风格 metrics 字典（ONNX/TRT 毫秒、参数量、GFLOPs）。

#### `print_table`（第 637～661 行）

- 拼 HTML 风格表头（CPU 信息、GPU 名）与 separator；逐 row LOGGER.info。