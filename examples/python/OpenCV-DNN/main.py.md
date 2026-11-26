# `examples/python/OpenCV-DNN/main.py`

本示例用 OpenCV `dnn.readNetFromONNX` 在无需 ONNX Runtime 的环境下跑 YOLO 导出模型。因 OpenCV 不读 Ultralytics 元数据，`imgsz` 与 `--task` 必须由 CLI 提供或通过输出张量形状启发式推断。支持检测、分割、姿态、OBB、end2-end 与网格头等布局；类名在无法从通道维推断时回退 COCO 80 类或数字 id。推理流程与 ONNXRuntime 示例相同地复用 `yolo_common` 预处理与后处理。

## 逐段说明

### 网络加载与任务推断（第 3–36 行）

`parse_args(..., imgsz=True)`；`readNetFromONNX`；方形 `imgsz`。`forward` 全部未连接输出层。根据 `outputs` 数量与张量维判断 `grid` 布局（`[1,C,A]` vs end2end `[1,300,attrs]`）。未指定 `task` 时：双输出→segment；grid 且 56 通道→pose 否则 detect；非 grid 则按最后一维 6/7 映射 detect/obb 否则 pose。据此构造 `names`（pose 固定 person；grid 从通道减固定偏移得 nc；e2e 对 detect/segment/rtdetr 用 COCO）。打印模型与任务。

### 推理链路（第 37–37 行）

读图、preprocess、`setInput`、`forward`、`postprocess`、`finish`。