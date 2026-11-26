# `ultralytics/models/sam/modules/__init__.py`

本文件位于 Ultralytics SAM 模型包的 `modules` 子目录，与其它具体实现文件（`sam.py`、`encoders.py`、`decoders.py` 等）同级。在工程结构上，它本可作为对外导出符号的入口，但当前实现未注册任何公开 API。SAM / SAM2 / SAM3 的组装与权重加载通常在上层 `build.py` 或预测器中完成，会直接按模块路径导入各类编码器与解码器。因此该文件在运行时对模型拓扑没有实质贡献，仅满足仓库许可证头规范。

## 逐段说明

### 许可证头（第 1 行）

仅包含 Ultralytics AGPL-3.0 许可证注释行，无 `import`、无类定义、无可执行逻辑。不包含 `__all__` 或其它导出声明。