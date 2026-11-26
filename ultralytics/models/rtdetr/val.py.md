# `ultralytics/models/rtdetr/val.py`

提供 `RTDETRDataset`（YOLO 数据集、`load_image` 直透父类）与 `RTDETRValidator`（构建 RT-DETR 验证集、`postprocess` 将归一化 query 输出转像素 xyxy 并按 conf/class 过滤）。解码器已在头内做 top-k，验证器不再重复 NMS 选 query。

## 逐段说明

### `RTDETRDataset`（行 16）

文档说明属性；无额外字段。

### `load_image(self, i, rect_mode=False)`（行 38）

调用 `super().load_image(i, rect_mode)` 返回图像与尺寸元组。

### `RTDETRValidator.build_dataset(self, img_path, mode="val", batch=None)`（行 83）

与 Trainer 类似构造 `RTDETRDataset`，`augment=False`、`rect=False`，`fraction` 用 `args.split` 或 `"val"`。

### `RTDETRValidator.postprocess(self, preds)`（行 113）

- **行 131–132**：list/tuple 取 `[0]`。
- **行 134**：split 为框、分数、标签。
- **行 135–136**：`xywh2xyxy` 后乘 `check_imgsz` 得 `(w,h,w,h)` 尺度；
- **行 137**：squeeze 分数与标签最后一维。
- **行 138–139**：指定 `classes` 时对非目标类分数 `masked_fill` 为 0。
- **行 140**：每张图 `score>conf` 的索引，截断 `max_det`。
- **行 142–145**：列表字典 `bboxes`/`conf`/`cls` 按 mask 索引切片。