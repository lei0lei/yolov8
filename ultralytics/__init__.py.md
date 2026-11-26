# `ultralytics/__init__.py`

本模块是 Ultralytics Python 包的公开入口。定义包版本号 `__version__`，在导入其他子模块前按默认将 `OMP_NUM_THREADS` 设为 `1` 以降低训练期 CPU 占用。从 `ultralytics.utils` 引入 `ASSETS`、`SETTINGS`（别名为 `settings`），以及 `checks`（`check_yolo`）与 `download`。通过 `MODELS` 与 `PLATFORM_EXPORTS` 元组声明可延迟加载的公开符号，并组装 `__all__`；在 Python 3.11 以下不把 Platform 相关名列入 `__all__`。`TYPE_CHECKING` 分支仅为静态类型检查器提供模型与 Platform 类型的提示。运行时通过 `__getattr__` 按名惰性导入 `ultralytics.models` 或 `ultralytics_platform` 中的类。`__dir__` 扩展 IDE 自动补全可见名集。

## 逐段说明

### 模块级环境与常量（第 3～31 行）

- 第 3 行：`__version__` 固定为当前发布版本字符串。
- 第 10～12 行：若环境未设置 `OMP_NUM_THREADS`，写入 `"1"`，避免 OpenMP 默认占满 CPU。
- 第 14～18 行：导入资产路径、全局设置、校验与下载工具；`settings` 指向 `SETTINGS` 单例。
- 第 20～21 行：`MODELS` 列出惰性导出的模型类名；`PLATFORM_EXPORTS` 列出云平台客户端与异常类名。
- 第 23～31 行：构建 `__all__`：版本、资产、全部模型名、条件性 Platform 符号、`checks`、`download`、`settings`。

### TYPE_CHECKING 导入块（第 33～36 行）

仅在类型检查运行时从 `ultralytics.models` 与 `ultralytics_platform` 导入符号，不改变运行时行为，供 mypy 等解析 `from ultralytics import YOLO` 等写法。

### `__getattr__`（第 39～47 行）

- 参数 `name` 为访问的缺失属性名。
- 若 `name in MODELS`：动态 `import_module("ultralytics.models")` 并 `getattr` 返回对应类。
- 若 `name in PLATFORM_EXPORTS`：Python `< 3.11` 时抛出 `ImportError` 说明需 3.11+；否则从 `ultralytics_platform` 惰性导入并返回。
- 否则抛出 `AttributeError`，表明模块无该属性。

### `__dir__`（第 50～52 行）

将 `globals()` 与 `MODELS`、`PLATFORM_EXPORTS` 合并去重排序，使 `dir(ultralytics)` 包含所有公开惰性名。

### `__main__` 入口（第 55～56 行）

脚本直接执行时仅打印 `__version__`，便于快速查看安装版本。