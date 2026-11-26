# `ultralytics/models/yolo/depth/calibrate.py`

深度头对数仿射标定 `d'=exp(a·log d + b)`：在验证集上“仅当有益才标定”的 CV 策略，供 `DepthTrainer.final_eval` 与 `Model.calibrate` 使用；不训练梯度、不改 decoder 权重。

## 逐段说明

### `_depth_head`（30–37）

解 DDP；取 `model.model` Sequential 末层；有 `cal_a` 才返回 head。

### `_rewind`（40–51）

InfiniteDataLoader 部分遍历后 `reset()` 回到 epoch 头，保证标定/绘图取“前 N 张”。

### `_delta1_none`（54–62）

应用 (a,b) 后 log 域 δ1：`|a·log_pred+b - log_gt| < log(1.25)` 比例。

### `select_calibration`（65–102）

候选 identity/current/scale-only(b=mean(lg-lp))；在 held-out log 像素上算 δ1；取最高 δ1，平局偏 identity。

### `select_calibration_cv`（105–155）

pairs 为每图 (log_pred,log_gt)；k 折；每折 fit/score 划分，`select_calibration` 得 scores；平均 cv_scores；非 identity 须超 margin；选 scale-only 时用全数据 refit b。

### `_collect_logpairs`（158–205）

标定前 head cal 置 1,0；遍历 dataloader 最多 max_images；有效 GT (0, max_depth) 且 pred finite；每图最多 20k 子采样；finally 恢复 cal_a/b。

### `fit_calibration_selective`（208–250）

无 head 或 pairs<2 则跳过；`select_calibration_cv`；写 head buffers；日志 CV 分数。

### `_plot_calibrated_batches`（253–300）

identity 下 raw 前向；列标题含 scale；`plot_depth_panels` 写 val_batch{ni}_calibrated.jpg；恢复 buffers。

### `calibrate_checkpoint`（303–376）

`torch_load`；deepcopy 浮点模型 `fit_calibration_selective`；同步 ema/model 的 cal；写 `depth_calibration` provenance；`torch.save`；可选 `_plot_calibrated_batches`。