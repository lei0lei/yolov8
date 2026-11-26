# `ultralytics/solutions/streamlit_inference.py`

`Inference` 类提供 Streamlit 侧栏选源（摄像头/视频/图片）、模型、conf/iou、类别与跟踪开关；加载 YOLO 后对视频循环或批量图片推理并双栏展示原图与标注图。CLI `yolo solutions inference` 通过 `subprocess streamlit run` 启动本文件。

## 逐段说明

### `__init__`（第 60～89 行）

检查 streamlit≥1.29；保存初始 conf/iou/imgsz/自定义 model 路径。

### `web_ui`（第 91～108 行）

隐藏菜单、HTML 标题与副标题。

### `sidebar`（第 110～131 行）

Logo；源选择；视频/摄像头时跟踪单选；conf/iou 滑条；非 image 时双列占位原图/结果。

### `source_upload`（第 133～156 行）

video 写临时 mp4；webcam id 0；image 多文件 tempfile 列表。

### `configure`（第 158～187 行）

模型下拉（yolo26 网格+自定义）；`YOLO` 加载；multiselect 类→`selected_ind`。

### `image_inference`（第 189～208 行）

逐上传图 predict+plot，删临时文件。

### `inference`（第 210 行起）

串联 UI；Start 后 image 走 `image_inference`，否则 `VideoCapture` 循环 predict/track、写 `ann_frame`/`org_frame`，Stop 按钮退出。