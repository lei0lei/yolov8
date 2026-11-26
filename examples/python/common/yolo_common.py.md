# `examples/python/common/yolo_common.py`

本模块为 `examples/python` 下 ONNXRuntime、OpenCV-DNN 等示例提供与后端无关的前后处理：CLI 参数、letterbox/拉伸/分类裁剪预处理、多任务后处理（检测网格与 e2e、分割 mask、姿态关键点、OBB、RT-DETR、深度、语义、分类）、OpenCV NMS 与 Ultralytics 调色板可视化。设计原则是后端只负责 `outputs = run(preprocess(...))`，解码与绘制集中在此，便于对比各推理运行时。

## 逐段说明

### 模块常量（第 15–28 行）

`SKELETON` 为 COCO 17 点连接；`COCO_NAMES` 为无元数据时的 80 类回退字典。

### `parse_args`（第 31–43 行）

统一 `--model/source/conf/iou/out/task/show`；可选 `--imgsz`。

### `preprocess`（第 46–69 行）

按 task 分支：分类矩形/方形裁剪、rtdetr/depth 拉伸、其余 letterbox 灰边 114。输出 NCHW RGB blob（可选 fp16）与 `(gain, left, top)` 映射参数。

### `_nms`（第 72–82 行）

轴对齐用 `NMSBoxesBatched`；旋转框按类分组 `NMSBoxesRotated`。

### `_masks`（第 85–95 行）

coef @ proto 得 mask，经 letterbox crop 与框内裁剪对齐原图布尔掩膜。

### `postprocess`（第 98–148 行）

**第一段（98–111）**：分类/深度/语义 early return。**第二段（113–141）**：rtdetr 归一化 xywh；end2end 与 grid 分支解析框与分数；grid 需 NMS；坐标反 letterbox。**第三段（142–148）**：segment 调 `_masks`；pose 重塑关键点并反变换。

### `_color`（第 157–160 行）

从十六进制调色板取 BGR。

### `draw`（第 163–196 行）

按 task 绘制 top5 分类、深度 colormap 叠加、语义 palette 混合、检测框/OBB 多边形/掩膜半透明/姿态骨架与点。

### `finish`（第 199–217 行）

打印摘要、`imwrite` 到 `--out`、可选 `imshow`。