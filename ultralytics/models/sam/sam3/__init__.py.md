# `ultralytics/models/sam/sam3/__init__.py`

SAM3 视觉-语言分割子目录的占位入口：仅包含 Meta 与 Ultralytics 版权声明，未在包加载时 re-export 具体模型类。SAM3 各模块（`sam3_image`、`encoder`、`decoder`、`vl_combiner` 等）由上层 SAM 构建逻辑或权重加载路径按名导入。保持空 `__init__` 可减少 `import ultralytics` 时对大型 SAM3 图的 eager 拉取。

## 逐段说明

### 行 1–4：版权与许可头

Ultralytics AGPL 头与 Meta Platforms 保留权利声明；无 Python 执行语句与符号定义。