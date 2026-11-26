# `examples/python/Region-Counter/main.py`

本示例结合 YOLO 跟踪与 Shapely 多边形区域，对穿过用户定义区域的物体检数，并支持鼠标拖拽平移计数区域。预置两个演示区域（多边形与矩形）；每帧绘制区域计数、轨迹折线与框标签。`run` 接受权重、设备、源视频、显示/保存与线宽等参数；`parse_opt`/`main` 提供 CLI。

## 逐段说明

### 模块级 `counting_regions`（第 19–39 行）

两个区域字典：`polygon`（Shapely）、`counts`、`dragging`、颜色与显示用文本色。

### `mouse_callback`（第 42–83 行）

按下：命中区域则记录拖动起点；移动：按偏移平移多边形顶点；抬起：清除 `dragging`。

### `run`（第 86–225 行）

**第一段**：校验源存在，加载模型到 cuda/cpu。**第二段**：视频读写与 `increment_path` 输出目录。**第三段**：循环 `track(persist=True)`，框中心在区域内则 `counts+=1`，画轨迹与区域标签。**第四段**：每帧末清零各区域计数；`q` 退出。

### `parse_opt`（第 228–242 行）

权重、device、必选 source、view/save、classes、各 thickness 参数。

### `main`（第 245–252 行）

`run(**vars(options))`。