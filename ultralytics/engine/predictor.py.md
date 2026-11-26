# `ultralytics/engine/predictor.py`

本模块实现推理管线基类 `BasePredictor` 及线程预取辅助函数 `_prefetch`，负责从任意源加载输入、预处理、经 `AutoBackend` 推理、任务相关后处理、可视化与落盘。对外接口为 `BasePredictor.__call__` / `stream_inference` / `predict_cli`，以及可被子类覆写的 `preprocess`、`inference`、`postprocess`、`setup_model` 等钩子。`Model.predict()` 通过 `task_map` 实例化各任务 `*Predictor`（如 `DetectionPredictor`）；CLI `yolo predict` 与 `yolo track`（跟踪在 predictor 上挂回调）均进入 `stream_inference`。模块顶部的 docstring 列举支持的 source 与导出格式，供命令行帮助阅读。

## 逐段说明

### 第 1～42 行：模块 docstring

说明 `yolo predict` 的 `source` 示例（摄像头、图、视频、目录、glob、YouTube、RTSP 等）与多种 `model=` 后缀格式，无执行逻辑。

### 第 44～66 行：导入

- **标准库**：`platform`（show 窗口）、`re`（解析帧号）、`threading`（推理锁）、`ThreadPoolExecutor`（预取）、`copy`/`deepcopy`、`Path`、`Callable`。
- **第三方**：`cv2` 读写与显示；`numpy` 堆叠；`torch` 张量与设备。
- **cfg**：`get_cfg`、`get_save_dir` 配置与输出目录。
- **data**：`load_inference_source` 统一数据源；`LetterBox` 缩放；`LoadImagesAndVideos` 判断是否预取。
- **nn.autobackend.AutoBackend**：多格式推理后端。
- **utils**：默认配置、日志、平台宏、回调、`colorstr`、`ops.Profile`；`check_imgsz`、`check_imshow`；`class_activation_map`；`attempt_compile`、`smart_inference_mode`。

### 第 68～78 行：`STREAM_WARNING`

多行字符串：未设 `stream=True` 时长视频/大列表结果会堆积在内存，附文档链接与 `stream=True` 用法示例。

### 第 81～91 行：`_prefetch`

单线程 `ThreadPoolExecutor`：`future=next(iterator)` 启动；循环 `future.result()` 得当前项，再提交下一个 `next`；`StopIteration` 结束生成器。用于 I/O 与 GPU 重叠。

### 第 94～138 行：`BasePredictor` 类文档与类属性

- `scale_fill=False`：是否拉伸至 `imgsz` 而非 letterbox。
- `default_iou=PREDICT_IOU`：预测默认 NMS IoU。

### 第 143～186 行：`BasePredictor.__init__`

`get_cfg(cfg, overrides)` → `self.args`；`get_save_dir` → `save_dir`。`conf` 默认 0.25；`iou` 默认 `default_iou`。`show` 经 `check_imshow`。占位字段：`model/device/imgsz/dataset` 等为 `None` 或空容器；`speed`/`pixels` 跑完后填；`callbacks` 默认或传入；`txt_path`；`_lock`；`add_integration_callbacks(self)` 挂 HUB 等集成回调。

### 第 188～210 行：`preprocess`

- **非 Tensor**：`pre_transform` → numpy；单张 `unsqueeze(0)`，多张 `np.stack`；`to(device)`；`BHWC→BCHW`；3 通道 `flip(1)` BGR→RGB；按 `model.fp16` 转 half/float 并 `/255` 归一化。
- **已是 Tensor**：`to(device, fp16/fp32)`，假定已 0～1。

返回 `(N,3,H,W)` 张量。

### 第 212～226 行：`inference`

`skip`：tensor 源、augment、embed 时不走 CAM。`visualize` 且 `base_model` 且非 skip：`class_activation_map(...)` 返回热力图路径等。否则 `self.model(im, augment=..., embed=..., **kwargs)` 标准前向。

### 第 228～247 行：`pre_transform`

统计 `im` 各图 shape 是否一致；构造 `LetterBox(imgsz, auto=同尺寸且rect且非scale_fill且(pt或dynamic非imx), scale_fill, stride)`；列表推导 letterbox 每张图。

### 第 249～251 行：`postprocess`（基类）

直接 `return preds`；子类实现 NMS、框缩放、掩码等。

### 第 253～271 行：`__call__`

设 `self.stream`；`stream=True` 返回 `stream_inference` 生成器；否则 `list(stream_inference(...))` 物化全部 `Results`。

### 第 273～292 行：`predict_cli`

创建 `stream_inference` 生成器，`for _ in gen: pass` 仅消费不保存引用，防止 CLI 长跑 OOM；注释禁止改此模式。

### 第 294～323 行：`setup_source`

若模型有固定 `imgsz` 且非 dynamic：`args.imgsz=model.imgsz`。`check_imgsz` 得 `self.imgsz`。`load_inference_source(source, batch, vid_stride, buffer, channels)` → `dataset`、`source_type`。流/截图/>1000 图/视频标志为真时懒导入 `torchvision`（NMS）；若 `stream` 属性为 False 则打 `STREAM_WARNING`。清空 `vid_writer`。

### 第 325～329 行：`_batches`

遍历 `batches` 每项设 `batch`/`dataset`，`on_predict_batch_start`，`yield from _batch_tiles`。

### 第 331～345 行：`_batch_tiles`

`args.tile` 假：原样 `yield batch, dataset, None`。真：`tile=True` 用 `imgsz`，否则 `args.tile` 为瓦片尺寸；`tile_batches` 切图；每次 yield 三元组（ paths/crops/info 子集, dataset, (原batch, windows, last) 上下文）。

### 第 347～349 行：`finalize_results`

基类恒等；子类可做源级后处理。

### 第 351～523 行：`stream_inference`（核心循环，分三段）

**段 A：锁内初始化（367～394）**

`with self._lock`：`cfg` 非空时若 project/name/save_dir/exist_ok 变则重算 `save_dir`，替换 `self.args`；补默认 `iou`；无 model 则 `setup_model`；非 `base_model` 时警告并关闭 augment/embed/visualize；`pt`+`end2end` 时 `set_head_attr(max_det, agnostic_nms)`；`setup_source(source or args.source)`；需保存则 `mkdir` labels 或 save_dir；重置 `seen/speed/pixels/windows`、`_bases`、`_sources`；三个 `Profile` 计时器；构造 `batches` 迭代器；CUDA 或 `prefetch` 且 `LoadImagesAndVideos` 多帧时用 `_prefetch` 包装。

**段 B：主循环（411～495）**

`tile` 与 tensor/embed/visualize 互斥则 `ValueError`。`on_predict_start`；维护 `tile_results`/`tile_windows`。对每个 batch：解析 `paths, im0s, s`；`preprocess`；首次 `warmup`；`inference`；若 `embed` 则 yield 张量并 `continue`；累计像素；`postprocess`；瓦片模式合并：`tile_context` 时累积至 `last` 为真，按图 `Results.merge_tiles`，再 `finalize_results`；`on_predict_postprocess_end`；对每张 `write_results` 更新 `seen`、per-result `speed`、`colors`；verbose/保存/show 时拼 `s[i]`；`StopIteration`（按 q 关窗）可 break；verbose 打印 `s`；`on_predict_batch_end`；`yield from self.results`。

**段 C：清理与汇总（496～523）**

`finally`：释放 `VideoWriter`；`batches.close()` 停预取；`dataset.close()` 停流线程。锁内：若 `seen>0` 算平均 ms 速度与 `pixels`；verbose 打 Speed 行；`destroyAllWindows`。锁外 `on_predict_end`（允许回调内再 predict）。

### 第 525～550 行：`setup_model`

`deepcopy(model)`；`AutoBackend(..., fp16=quantize==16, fuse=True, end2end=nms is False)`；`self.device=model.device`；CPU 上 onnx/openvino 限制 OpenMP 线程；`eval()`；`attempt_compile`。`smart_inference_mode(False)` 允许此处图变换。

### 第 552～615 行：`write_results`

单张 im 扩 batch 维。按 `source_type` 拼日志前缀与 `frame`（流/img/tensor 用 count；截图用 dataset.frame；视频从 `s[i]` 正则取帧号）。`key` 视频用路径否则 batch 索引；`base` 去重 stem（`-2` 后缀防覆盖）。`txt_path` 在 labels 下。拼分辨率、verbose、推理 ms。`save/show/save_frames` 时 `result.plot(...)` → `plotted_img`。`save_txt` 先删旧 txt 再写；`save_crop`；`show`；`save`/`save_frames` 调 `save_predicted_images`。返回日志字符串。

### 第 617～649 行：`save_predicted_images`

流/视频：`fps` 经 `vid_stride`；`save_frames` 写独立 jpg 帧目录；`save` 时懒建 `VideoWriter`（平台相关后缀与 fourcc），`write` 帧。纯图像：`cv2.imwrite` 为 jpg。

### 第 651～671 行：`show`

Linux/Windows 首次 `namedWindow`+按比例缩窗（tkinter 读屏 0.9）；`imshow`+`waitKey`（图 300ms，视频 1ms）；`q` 抛 `StopIteration` 终止流。

### 第 673～676 行：`run_callbacks`

遍历 `callbacks[event]`，传入 `self`。

### 第 678～680 行：`add_callback`

向 event 列表 append 函数。