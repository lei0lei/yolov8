# `ultralytics/solutions/ocr.py`

OCR 管线：YOLO 文本检测（可 seg/OBB/水平框）→ 页面旋转与行分组 → `TextRecognizer`（ONNX 或 TensorRT PP-OCRv6）逐词识别 → 按阅读顺序标注。

## 逐段说明

### 类 `TextRecognizer`

- `__init__`（36～67）：CUDA 可选 TensorRT 引擎（带锁）或 ONNXRuntime IOBinding；解析 CTC 字符表。
- `load_engine`（69～98）：按 TRT 版本与 SM 缓存 engine，无则从 ONNX FP16 动态 shape 构建。
- `__call__`（100～137）：竖长 crop 旋转；按宽排序 batch；右 pad 到 batch 最大宽；推理后 CTC 去 blank/重复得文本与均分置信度。

### 类 `OCR`

- `__init__`（149～163）：`tracker=None`；默认大模型/imgsz/max_det；`recognizer` 字符串或 callable。
- `lines`（166～191）：按 y 带宽分 band，同 band 按 x 裂行（间距>1.5h）。
- `read_words`（194～250）：估计页方向，两向 0/π 旋转试读，用 8 最宽词置信度选优；批量识别 rest；输出含 line 号的词条列表。
- `process`（252～281）：动态 imgsz 不放大；构建多边形；`read_words`；合并检测置信度；`box_label` 绘制；`SolutionResults.texts`。