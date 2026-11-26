# `docs/model_data.py`

本模块维护文档站模型对比页使用的静态元数据字典 `data`，按模型系列（YOLO26、YOLO11、YOLOv8、RT-DETR、第三方检测器等）记录作者、机构、日期、论文/仓库/文档链接，以及各尺寸变体的 mAP、参数量、FLOPs、CPU 与 T4 延迟等指标。FLOPs 与官网表格对齐为单一事实来源。直接执行脚本时把 `data` 序列化为当前工作目录下的 `model_data.json`，供前端图表或宏渲染消费；模块导入时不产生副作用。

## 逐段说明

### 模块文档字符串与 `data` 字典（第 2–349 行）

顶层键为公开模型族名称。每个条目固定字段：`author`、`org`、`date`、`arxiv`（可 None）、`github`、`docs`、`performance`。`performance` 子键为变体代号（如 `n/s/m/l/x`），值为含 `size`、`map`、`cpu`、`t4`、`params`、`flops` 的字典；部分遗留条目 `cpu` 为空字符串表示未填报。涵盖 Ultralytics 自有系列、YOLOv5–v10、竞品（PP-YOLOE+、DAMO-YOLO、Gold-YOLO、D-FINE、YOLO-World、RTMDet、YOLO-NAS、FCOS、SSD、LWDETR 等）及 RT-DETR 变体。

### `if __name__ == "__main__"`（第 351–356 行）

`json.dump(data)` 写入 `model_data.json`，供 `build_docs.py` 或发布流水线拉取最新对比数据。