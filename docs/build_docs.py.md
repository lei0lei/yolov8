# `docs/build_docs.py`

本脚本是本地完整文档构建入口：备份 `docs/en` 与 `docs/macros`，克隆 ultralytics/docs 比较页、为全站 md 补 frontmatter、调用 `build_reference_docs` 生成 API 参考、用 MiniJinja 展开宏，再 `zensical build --strict` 产出 `site/`，最后无论成败都恢复备份树。避免污染 Git 跟踪的英文源与宏文件；比较页与完整 reference 渲染仅存在于构建临时态。依赖 dev extra 中的 zensical 与 minijinja。

## 逐段说明

### `prepare_docs_markdown`（第 24–41 行）

删除旧 `site/` 与 `docs/repos`；浅克隆 docs 仓库 main 分支；用其 `en/compare` 覆盖本地 compare；对 `docs/en` 全部 md  tqdm 调用 `update_markdown_files`。

### `update_markdown_files`（第 44–81 行）

读文件、统一弯引号为 ASCII 撇号；无 frontmatter 则插入标准 SEO 块；为 `=== ` 内容标签前后补空行；末尾补换行并写回。

### `render_jinja_macros`（第 84–178 行）

从 `mkdocs.yml` 与 `default.yaml` 装载上下文（含 SafeFallbackLoader 忽略未知 YAML 标签）。MiniJinja 环境指向 `docs/en` 与 `docs`，注册 `indent` 过滤器。跳过 `macros/` 与 `reference/` 路径；含 `{{`/`{%` 的页面分离 frontmatter 后渲染正文，计数宏命中并写回。

### `backup_docs_sources` / `restore_docs_sources`（第 181–201 行）

在仓库父目录建临时目录，复制 `en` 与 `macros`；恢复时删原目录再从备份 copytree。

### `main`（第 204–243 行）

缺 zensical 则退出。try 块：备份→prepare→`build_reference_docs(update_nav=False)`→render macros→删 repos→zensical strict build→restore。finally 确保未 restore 时仍恢复并清理 repos。