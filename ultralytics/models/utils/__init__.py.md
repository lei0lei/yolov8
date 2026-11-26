# `ultralytics/models/utils/__init__.py`

该文件仅保留 Ultralytics AGPL 许可证头，不向包外再导出符号。RT-DETR / DETR 相关工具（匈牙利匹配、`get_cdn_group`、损失类）由调用方直接从同级模块 `ops.py`、`loss.py` 导入，本 `__init__.py` 不承担聚合职责，也不改变 `models` 包的懒加载行为。

## 逐段说明

### 行 1–2：许可证注释

标准 Ultralytics 许可证与项目链接注释，无运行时代码。