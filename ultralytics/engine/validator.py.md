# `ultralytics/engine/validator.py`

本模块定义验证流程基类 `BaseValidator`，在独立 `model.val()` 或训练周期内评估模型：构建/复用 dataloader、`AutoBackend` 推理、可选训练时 loss 累计、指标更新与 DDP 聚合。对外入口为 `__call__(trainer=None, model=None)`，子类（如 `DetectionValidator`）实现 `get_dataloader`、`postprocess`、`init_metrics`、`update_metrics` 等。`Model.val()` 与 `Model.calibrate()` 通过 `task_map["validator"]` 构造具体类；`Trainer` 在验证阶段传入 `trainer` 走 EMA 与 loss 分支。

## 逐段说明

### 第 1～33 行：模块 docstring

`yolo val` 用法与支持的导出模型后缀列表，与 predictor 模块对称。

### 第 35～59 行：导入

- **标准库**：`json` 写 predictions.json；`time` 绘图时间戳；`deepcopy`；`Path`。
- **第三方**：`numpy`；`torch`；`torch.distributed` DDP reduce。
- **cfg**：`get_cfg`、`get_save_dir`。
- **data.utils**：`check_cls_dataset`、`check_det_dataset`、`convert_ndjson_to_yolo_if_needed`。
- **AutoBackend**：独立验证加载权重。
- **utils**：`LOCAL_RANK`、`RANK`、`TQDM`、回调、`emojis`；`check_imgsz`；`Profile`、`linear_sum_assignment`；`attempt_compile`、`autocast`、设备后端、`smart_inference_mode`、`torch_distributed_zero_first`、`unwrap_model`。

### 第 62～113 行：`BaseValidator` 文档字符串

描述属性（args、dataloader、model、data、device、training、stats、iouv、jdict、speed 等）与可覆盖方法清单。

### 第 115～151 行：`BaseValidator.__init__`

懒导入 `torchvision`（避免 postprocess 计时污染）。`get_cfg(overrides=args)`；大量字段初值为 `None` 或默认 `speed` 字典；`save_dir` 或 `get_save_dir`；按 `save_txt` 建 `labels` 或根目录；`conf` 默认 OBB 0.01 否则 0.001；`iou` 默认 0.7；`check_imgsz(..., max_dim=1)` 写回 `args.imgsz`；空 `plots`；默认或传入 `callbacks`。

### 第 153～157 行：属性 `num_images`

`tile` 模式用 `len(dataset.tile_counts)`，否则 `len(dataset)`，供速度按图平均。

### 第 159～218 行：`merge_tiles`

瓦片验证合并逻辑（`args.tile` 时由 `__call__` 生成多路输出）。

- 遍历 `preds` 与 `batch["tile"]`、`batch["source_index"]` 对齐。
- **dict 预测**：`scale_preds`（子类实现）缩放到瓦片在原图中的窗口 `shape`。
- **classify**：包成 `{"probs": pred}`。
- **semantic**：`ops.scale_masks` 最近邻还原语义 mask。
- **其他（如 depth）**：双线性 `interpolate` 到 `shape`。
- 按 `source` 累积 `predictions`/`windows`；未满 `tile_counts[source]` 则 `continue`。
- 取整图 `get_image_and_label(..., full=True)`；非 classify 走 transforms 最后一项；`preprocess_targets(collate_fn([sample]))` 得 `full_batch`。
- `ops.merge_tiles` 合并检测/分割等到整图尺寸；有 `masks` 转 `Masks.data`；输出格式统一为列表或 depth 张量等。
- `del pending[source]`；`yield output, full_batch, source`。

### 第 220～393 行：`__call__`（验证主流程，分多段）

**段 A：模式与模型准备（232～312）**

`training = trainer is not None`。非训练且 DDP：`torch_distributed_zero_first` 内 `convert_ndjson_to_yolo_if_needed`。`get_model(model, trainer)` → 实际推理模块。`augment = args.augment and not training`。

- **训练中**：同步 `end2end` 与 `nms`；`device`/`data` 来自 trainer；AMP 且非 CPU 时 `quantize=16`，`model.float()`；初始化 `loss` 为零张量字典；`plots` 仅可能的 early-stop 轮或最后一 epoch；`model.eval()`。

- **独立验证**：未训练 YAML 警告；`add_integration_callbacks`；按 device 建 `AutoBackend`（DDP rank 用当前设备）；`quantize` 与实际 fp16 对齐；`stride`/`format`；非 pt 且固定 `imgsz` 用导出尺寸；非 pt/torchscript 且非 dynamic 时 batch 取自 metadata 并日志；按 task `check_cls_dataset` 或 `check_det_dataset`，否则 `FileNotFoundError`；CPU/MPS `workers=0`；非 pt/dynamic 关 `rect`；`stride`、`names`、`dataloader` 懒建；`eval`；可选 `compile`；`warmup`。

**段 B：批循环（314～355）**

`on_val_start`；四个 `Profile`；`TQDM`；`init_metrics(unwrap_model(model))`；`jdict=[]`；`tile_results={}`。

每 batch：`on_val_batch_start`；`batch_i`；`preprocess`；`autocast` 下 `model(batch["img"], augment)`；训练中 `loss` 累加 `model.loss` 分项；`postprocess`；`tile` 则 `merge_tiles` 展开为多 `(output, targets, image_i)`，否则单组；`update_metrics`；前 3 批主进程可 `plot_val_samples`/`plot_predictions`；`on_val_batch_end`。

**段 C：收尾（357～393）**

`gather_stats()`；主进程：`get_stats()`、按 `num_images` 填 `speed`、`finalize_metrics`、`print_results`、`on_val_end`。

- **训练返回**：`loss` 与 batch 数 stack，`dist.reduce` 到 0；`RANK>0` 返回 `None`；主进程 `loss/len(dataloader)` 与 `stats`、trainer `label_loss_items` 合并，五位小数 dict。

- **独立返回**：`RANK>0` 可早返 `stats`；打 Speed 四段 ms；`save_json` 写 `predictions.json` 并 `eval_json`；有 plots/json 时日志 save_dir；返回 `stats`。

### 第 395～432 行：`match_predictions`

输入：按置信降序的预测类 `(N,)`、GT 类 `(M,)`、IoU 矩阵 `(M,N)`、`use_scipy` 是否匈牙利匹配。

- `correct` 为 `(N, T)` bool，`T=len(iouv)`（默认 10 档 0.5～0.95）。
- `correct_class` 掩码：类不对则 IoU 置零。
- IoU/thresholds 转 numpy。

**scipy 分支**：每阈值 `cost_matrix = iou*(iou>=thr)`；`linear_sum_assignment(-cost)` 最大化 IoU；有效配对标 `correct[det_idx, i]=True`。

**贪心分支**：`matched` 记录每 GT 在各阈值是否已被占；按检测 j 降序；`available` 屏蔽已匹配 GT；每阈值取 argmax GT；若 IoU≥阈值则标对并更新 `matched`。

返回 `torch.from_numpy(correct)`。

### 第 434～445 行：`get_model`

有 `trainer` 返回 `trainer.ema.ema`；否则 `deepcopy(model)`，避免污染原权重。

### 第 447～454 行：回调

`add_callback` append；`run_callbacks` 遍历执行。

### 第 456～462 行：`get_dataloader` / `build_dataset`

基类 `NotImplementedError`，子类按 task 构建 `YOLODataset` 等。

### 第 464～473 行：`preprocess` / `preprocess_targets`

`preprocess` 默认原样 batch。`preprocess_targets`：除 `img` 外 tensor 字段 `.to(device, non_blocking=...)`，整图评估时大图可留 CPU。

### 第 475～477 行：`postprocess`

基类恒等；子类做 NMS、尺度还原等。

### 第 479～504 行：指标钩子（空实现）

`init_metrics`、`update_metrics`、`finalize_metrics`、`get_stats`（返回 `{}`）、`gather_stats`、`print_results`、`get_desc` 由子类填充；`metric_keys` 属性默认 `[]`。

### 第 506～508 行：`on_plot`

`plots[Path(name)] = {data, timestamp}` 登记可视化产物。

### 第 510～521 行：绘图与 JSON 钩子

`plot_val_samples`、`plot_predictions`、`pred_to_json`、`eval_json` 基类空体，供 detect/segment 等覆盖。