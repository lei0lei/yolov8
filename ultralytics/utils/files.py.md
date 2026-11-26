# `ultralytics/utils/files.py`

本模块汇集训练与导出常用的路径与文件操作：临时切换工作目录、处理含空格路径、自动递增 run 目录名、文件元信息查询、在 glibc 可执行栈限制下导入特定 native 模块、查找最近 `last*.pt` 以及批量将权重转 FP16 保存。`WorkingDirectory` 同时可作装饰器与上下文管理器；`spaces_in_path` 在含空格时将资源复制到临时无空格路径并在退出时写回。

## 逐段说明

### 类 `WorkingDirectory`（第 18～56 行）

#### `__init__`（第 45～48 行）

- 保存目标目录 `self.dir` 与进入前 `Path.cwd().resolve()` 为 `self.cwd`。

#### `__enter__`（第 50～52 行）

- `os.chdir(self.dir)`。

#### `__exit__`（第 54～56 行）

- 无论异常与否 `os.chdir(self.cwd)` 恢复。

### `spaces_in_path`（第 59～106 行）

- **第 78 行**：路径字符串无空格则 **第 105～106 行** 直接 `yield path`。
- **第 79～80 行**：记录是否原为 str，转为 `Path`。
- **第 83～91 行**：`TemporaryDirectory` 下 `tmp_path` 名中空格换 `_`；目录 `copytree` 或文件 `copy2`。
- **第 93～95 行**：`yield` 临时路径（原类型 str 则转 str）。
- **第 97～102 行**：`finally` 将临时内容拷回原路径（目录 `dirs_exist_ok=True`）。
- **第 104～106 行**：无空格分支 `yield path`。

### `increment_path`（第 109～157 行）

- **第 143 行**：`Path(path)` 规范化。
- **第 144～152 行**：路径存在且非 `exist_ok`：文件则分离 suffix，循环 `n in range(2,9999)` 拼 `{path}{sep}{n}{suffix}` 直至不存在。
- **第 154～155 行**：`mkdir=True` 时 `mkdir(parents=True, exist_ok=True)`。
- **第 157 行**：返回 `Path`。

### `file_age`（第 160～163 行）

- 用本地时区 now 与文件 mtime 差值，返回整天数 `dt.days`。

### `file_date`（第 166～169 行）

- mtime 转本地时区，格式 `YYYY-M-D`（月日不补零）。

### `file_size`（第 172～181 行）

- `path` 非 str/Path 或缺失返回 `0.0`。
- 文件：`st_size / (1<<20)` MiB。
- 目录：递归 `**/*` 文件大小之和 MiB。

### `import_execstack`（第 184～211 行）

- **第 194～198 行**：正常 `importlib.import_module`；`ImportError` 消息不含可执行栈文案则 re-raise。
- **第 199 行**：在 site-packages 下按 patterns glob 共享对象。
- **第 200～208 行**：读 ELF 头解析 program header，对 `PT_GNU_STACK` 且带执行标志的节清除 PF_X 位。
- **第 209～210 行**：删除已部分加载的 `{name}.*` 子模块缓存。
- **第 211 行**：再次 import 并返回模块。

### `get_latest_run`（第 214～217 行）

- `glob **/last*.pt`，按 mtime 取最大路径，无则 `""`。

### `update_models`（第 220～257 行）

- **第 235～240 行**：懒 import YOLO、default_class_names、LOGGER；`updated_models` 子目录。
- **第 242～257 行**：对每个 `model_name`：`YOLO` 加载、`half()`；`update_names` 时 names 换 coco8；`model.save` 到目标路径。