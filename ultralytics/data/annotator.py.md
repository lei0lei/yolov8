# `ultralytics/data/annotator.py`

本模块提供一条「检测框 → 分割多边形」的自动标注流水线：先用 YOLO 检测，再用 SAM 按框生成分割轮廓，并写出 YOLO 分割格式的 `.txt` 标签。对外唯一公开入口是顶层函数 `auto_annotate`（L10–L67）。典型调用方为脚本、Notebook 或文档示例中的 `from ultralytics.data.annotator import auto_annotate`；与 `converter.yolo_bbox2segment` 类似，但面向任意图片目录且输出目录可自定义。模块仅依赖 `ultralytics.YOLO`、`ultralytics.SAM` 与标准库 `pathlib`，不接触数据集缓存或 YAML 校验逻辑。

## 逐段说明

### 导入区（L1–L7）

- **L1**：Ultralytics 许可证头，不参与运行逻辑。
- **L3 `from __future__ import annotations`**：推迟注解求值，便于类型标注使用 `str | Path` 等写法。
- **L5 `pathlib.Path`**：统一把 `data`、`output_dir` 解析为路径对象。
- **L7 `from ultralytics import SAM, YOLO`**：懒加载高层推理 API；`auto_annotate` 内分别实例化检测与分割模型。

### `auto_annotate`（L10–L67）

**职责**：对 `data` 目录下图像流式检测，对每个有框的结果调用 SAM，将归一化多边形写入 `output_dir` 中与源图同名的 `.txt`。

#### 参数与模型初始化（L44–L50）

- **L44–L45**：用字符串权重路径构造 `YOLO` 与 `SAM` 实例；权重名如 `yolo26x.pt`、`sam_b.pt` 会触发自动下载（由 Ultralytics 下载逻辑处理）。
- **L47**：`data` 转为 `Path`。
- **L48–L49**：若未指定 `output_dir`，默认为 `data` 父目录下 `{data.stem}_auto_annotate_labels`，与输入文件夹并列。
- **L50**：创建输出目录（`parents=True`，已存在不报错）。

#### 检测流式推理（L52–L54）

- **L52–L54**：`det_model(..., stream=True, ...)` 返回可迭代结果生成器，逐张处理，避免一次性加载全部结果。
- 传入 `conf`、`iou`、`imgsz`、`max_det`、`classes` 直接映射到 YOLO `predict`；`device=""` 表示自动选设备。

#### 主循环：检测 → SAM → 写标签（L56–L67）

- **L56 `for result in det_results`**：每张图一个 `Results` 对象。
- **L57**：初始化 `segments` 为空列表。
- **L58 `if class_ids := result.boxes.cls.int().tolist()`**：
  - 若无检测框，`class_ids` 为空，整个 `if` 跳过；仍会打开输出 `.txt` 并写入空文件（见 L63–L67）。
- **L59**：取像素坐标 `xyxy` 框，供 SAM 使用。
- **L60**：`sam_model(result.orig_img, bboxes=boxes, ...)[0]` 在原始分辨率图像上按框分割；`verbose=False, save=False` 不写可视化。
- **L61**：`sam_result.masks.xyn` 为归一化多边形列表（每实例一个 `(n,2)`）；`class_ids` 按 SAM 返回的 `boxes.cls` 索引重排，使分割与 SAM 保留的实例一一对应。

**写文件（L63–L67）**

- **L63**：输出路径为 `output_dir / det_model.predictor.txt_path.name + ".txt"`。`txt_path.name` 来自检测 predictor 当前帧的标签基名（与输入图像 stem 一致）。
- **L64–L67**：遍历 `segments` 与 `class_ids`：
  - **L65**：`len(s) >= 3` 才写入；少于 3 点不是合法多边形，与数据加载器要求一致。
  - **L66–L67**：每行格式为 `class_id x1 y1 x2 y2 ...`（归一化浮点，空格分隔），即 YOLO 分割标签行。

#### 标签格式（输入 / 输出）

| 阶段 | 格式 |
|------|------|
| 输入 | 目录内任意 YOLO 可读的图像；无标签文件 |
| 中间 | 检测：`xyxy` 像素框；SAM：`masks.xyn` 归一化顶点 |
| 输出 `.txt` | 每行：`cls` + 偶数个归一化坐标（≥6 个数，即 ≥3 点） |

**调用关系**：不直接被 `engine/trainer` 引用；面向用户工具链。与 `converter.yolo_bbox2segment` 区别：本函数不写回 `labels-segment` 固定结构，且检测与 SAM 类 ID 对齐方式相同（经 SAM 的 `boxes.cls` 过滤）。