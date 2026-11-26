# `docs/build_reference.py`

本脚本遍历 `ultralytics/` 包，为公开 API 生成 `docs/en/reference/` 下 Markdown：支持仅 `:::` 占位 stub 流程与完整 Google docstring 渲染（签名表、摘要标签页、源码折叠面板）。可更新 `mkdocs.yml` 的 Reference 导航；缺类型注解时 `build_reference_docs` 抛错阻断发布。

## 逐段说明

### 数据类（第 56–115 行）

`ParameterDoc`、`ReturnDoc`、`ParsedDocstring`、`DocItem`、`DocumentedModule` 承载结构化文档模型。

### `extract_classes_and_functions`（第 123–128 行）

正则提取顶层 class/function 名（占位流程）。

### `_with_reference_title` / `_existing_frontmatter` / `create_placeholder_markdown`（第 131–186 行）

注入 SEO `title`；保留已有 frontmatter；写 `## :::` 桩与贡献 admonition。

### AST 辅助（第 189–346 行）

`_get_source`、注解/默认值格式化、`collect_signature_parameters`、`format_signature`（含长签名折行）。

### Docstring 解析（第 349–528 行）

`_split_section_entries`、`_parse_named_entries`、`_parse_returns`、`parse_google_docstring`、`merge_docstrings`（类与 `__init__` 合并）。

### `_should_document` 与源码片段（第 531–568 行）

过滤私有符号；`_collect_source_block`、`_get_definition_signature`。

### `parse_function` / `parse_class` / `parse_module`（第 571–723 行）

**`parse_function`**：方法/属性/异步识别。**`parse_class`**：MRO 继承 `__init__`、方法子节点。**`parse_module`**：模块级类与函数列表。

### 渲染层（第 726–1082 行）

`_render_table`、`_merge_params`、`render_docstring`（Args/Returns/Examples admonition 等）、`render_summary_tabs`（MkDocs 摘要标签）、`render_item`（类含 Methods 表与 children）、`render_module_markdown`。

### `create_markdown`（第 1085–1112 行）

写完整参考页；新文件 `git add -f`。

### 导航（第 1115–1225 行）

`nested_dict`、`create_nav_menu_yaml`、`extract_document_paths`、`update_mkdocs_file`（路径集变化才写，可选 prettier）。

### 构建入口（第 1228–1324 行）

**`build_reference_placeholders`**：删 orphan stub，更新 nav。**`build_reference_docs`**：全量 docstring 渲染，缺类型则 `ValueError`。**`build_reference_for`**：临时切换包根。**`main`**：默认 placeholder + `update_nav=True`。