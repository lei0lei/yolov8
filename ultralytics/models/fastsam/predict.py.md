# `ultralytics/models/fastsam/predict.py`

`FastSAMPredictor` 继承分割 `SegmentationPredictor`，在 `finalize_results` 中修正全图框并调用 `prompt` 按用户提示筛选 mask。`prompt` 支持框（IoU 与面积比）、点（前景/背景窗口命中）与文本（裁剪 mask 区域后 MobileCLIP 相似度）。类属性 `default_iou=0.7` 针对嵌套单类 mask 的 NMS，与 YOLO 默认不同。

## 逐段说明

### 行 11–17：导入

分割 Predictor、线程池、张量与 PIL、`box_iou`、`clip_boxes`/`clip_coords`、`TORCH_1_10`、`adjust_bboxes_to_image_border`。

### 类属性 `default_iou = 0.7`（行 36）

注释说明用于嵌套 mask 的 IoU 阈值。

### `finalize_results(self, results)`（行 38）

对每个 `result`：
- **行 41–43**：构造覆盖整图的 `full_box`（xyxy，设备与 dtype 与 `result.boxes` 一致）。
- **行 44**：`adjust_bboxes_to_image_border` 修正框贴边。
- **行 45–47**：`box_iou(full_box[None], boxes)`，IoU>0.9 的索引将对应框设为 `full_box`（整图候选）。
- **行 49**：返回 `self.prompt(results, **getattr(self.args, "prompts", {}))`，每轮 predict 由 Model 写入 `prompts`。

### `prompt(self, results, bboxes=None, points=None, labels=None, texts=None)`（行 51）

- **行 64–65**：三者皆为 None 时直接返回 `results`。
- **行 66–68**：统一为 list；遍历每个 `result`。

对单张 `result`：
- **行 70–72**：空结果原样加入 `prompt_results`。
- **行 73**：`crops = result.masks.crops(orig_shape)`，每 mask 及其框内的裁切。

**嵌套 `window(crop, x1, y1, x2, y2)`（行 75–78）**：按框窗口从 mask 张量切片（相对 crop 框坐标换算）。

**行 81**：`idx` 为与 `result` 等长的 bool 掩码，初值为全 False。

**bboxes 分支（行 82–91）**：
- 张量化、必要时扩维、 `clip_boxes` 到原图；
- 计算 prompt 框面积与每个 mask 在框内面积、mask 全面积；
- `union = bbox_area + full_mask_area - mask_in_bbox`；
- `mask_areas/union` 每 prompt 取 argmax，且在至少一列有正覆盖时置 `idx` 对应为 True。

**points 分支（行 92–109）**：
- 坐标 int64、`clip_coords`（上界为 shape-1）；
- `labels` 缺省全 1；
- 断言 labels 与 points 长度一致；
- `point_masks`：各点 1×1 窗口是否命中 mask；
- 若 labels 全 0：`point_idx` 全 True；否则前景点任一为真；
- 再与「背景点未命中任一 mask」相与；
- `idx |= point_idx`。

**texts 分支（行 110–127）**：
- 单字符串转 list；
- 遍历框与 crop：面积过小（torch1.9 用 `sum(0).sum()`）则跳过；
- 从原图裁 crop，将 mask 窗口贴入全零通道后与 crop 相乘，转 PIL（RGB）；
- 若有有效 crop：`similarity = _clip_inference(crop_ims, texts)`，对每个文本 argmax 对应 `keep` 索引置 `idx`。

**行 129**：`prompt_results.append(result[idx])` 布尔索引筛选。

**行 131**：返回 `prompt_results`。

### `_clip_inference(self, images, texts)`（行 133）

- **行 144–147**：懒加载 `MobileCLIPTS` 到 `self.clip`。
- **行 148–153**：CUDA 且图够多时 `ThreadPool` 并行 `image_preprocess`，否则逐张处理并 `stack`。
- **行 154–156**：`encode_image`、`encode_text(tokenize(texts))`，返回 `text_features @ image_features.T` 形状 `(M,N)`。