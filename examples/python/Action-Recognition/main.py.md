# `examples/python/Action-Recognition/main.py`

本示例先用 YOLO 跟踪视频中的行人，再对每个 track 收集短时帧裁剪序列，送入 TorchVision 视频模型或 Hugging Face XCLIP 类模型做动作识别，并把 Top 标签叠在画面上。支持 YouTube URL（经 `get_best_youtube_url`）与本地 mp4。分类器路径由模型名是否在 TorchVision 注册表内自动分支。

## 逐段说明

### `TorchVisionVideoClassifier`（第 21–153 行）

**`__init__`（73–84）**：校验 `model_name`，加载权重到 `select_device`。**`available_model_names`（87–93）**：返回键列表。**`preprocess_crops_for_video_cls`（95–119）**：v2 变换后堆叠为 `(1,C,T,H,W)`。**`__call__`（121–133）**：`inference_mode` 前向。**`postprocess`（135–153）**：argmax 类别名与 softmax 置信度。

### `HuggingFaceVideoClassifier`（第 156–280 行）

**`__init__`（183–205）**：加载 processor/model，可选 fp16。**`preprocess_crops_for_video_cls`（207–237）**：归一化 resize 为 `(1,T,C,H,W)`。**`__call__`（239–255）**：文本 token 与 pixel_values 前向得 `logits_per_video`。**`postprocess`（257–280）**：每样本 top-2 标签与概率。

### `crop_and_pad`（第 283–311 行）

按框加 margin 裁剪，再取正方形区域并 resize 224×224。

### `run`（第 314–463 行）

选分类器；解析 YouTube；`VideoCapture`/`VideoWriter`；按 `skip_frame` 积累每 track 裁剪序列；满 `num_video_sequence_samples` 时批量推理；按重叠率节流更新；`Annotator.box_label` 显示双行 top 标签；`imshow` 与 `q` 退出。

### `parse_opt` / `main`（第 466–509 行）

CLI 暴露权重、设备、源、输出、裁剪/序列/跳过/重叠/fp16/分类器与零样本 labels；`main` 调 `run`。