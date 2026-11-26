# `ultralytics/models/rtdetr/predict.py`

`RTDETRPredictor` 继承 `BasePredictor`，类属性 `scale_fill=True` 表示推理缩放为填充方形。`postprocess` 将解码器已 top-k 的归一化 xywh+score+cls 张量过滤置信度与类别，转 xyxy 并按原图宽高缩放，封装为 `Results`。

## 逐段说明

### 行 5–7：导入

`BasePredictor`、`Results`、`ops`。

### 类属性 `scale_fill = True`（行 33）

推理预处理使用 scale-fill 方形策略。

### `postprocess(self, preds, img, orig_imgs)`（行 35）

- **行 52–53**：`preds` 为 list/tuple 时取首元素。
- **行 54**：最后一维按 `(4,1,1)` 切为 `bboxes`、`scores`、`labels`。
- **行 55–56**：`orig_imgs` 非 list 时 `convert_torch2numpy_batch` 并 BGR→RGB 通道逆序。
- **行 58–68**：逐 batch 元组 zip：
  - `bbox = ops.xywh2xyxy(bbox)`；
  - `idx = score.squeeze(-1) > self.args.conf`；
  - 若 `self.args.classes` 非 None：`idx` 再与类别 membership 相与；
  - `pred = cat([bbox,score,label],-1)[idx][:max_det]`；
  - x 坐标乘 `ow`，y 乘 `oh`；
  - 追加 `Results(orig_img, path, names, boxes=pred)`。
- **行 69**：返回 `results` 列表。