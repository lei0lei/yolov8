# `ultralytics/cfg/__init__.py`

Ultralytics CLI 与配置中枢：定义 `TASKS`/`MODES`/默认数据与模型映射、`SOLUTION_MAP`、量化与参数类型校验集合；`get_cfg` 合并 YAML 与 overrides；`entrypoint` 解析 `yolo` 子命令并调度 train/val/predict 等；`handle_yolo_solutions` 驱动 Vision Solutions 流水线。

## 逐段说明

### 模块常量（第 40～317 行）

`SOLUTION_MAP` 名称→类名；`MODES`/`TASKS` 及 `TASK2DATA`/`TASK2MODEL`/`TASK2METRIC`；`CLI_HELP_MSG`/`SOLUTIONS_HELP_MSG`；`QUANTIZE_ALIASES`；`CFG_*_KEYS` 用于 `check_cfg`。

### `cfg2dict`（第 320～351 行）

路径 YAML 加载、SimpleNamespace→dict、已是 dict 则原样。

### `get_cfg`（第 354～398 行）

`_handle_deprecation`；overrides 对齐合并；`project`/`name` 数值转 str；`name=model` 用权重 stem；`check_cfg`；返回 `IterableSimpleNamespace`。

### `check_cfg`（第 401～544 行）

遍历项：numpy 标量 `.item()`；None 非法检测；分 float/fraction/int/bool/str、`scale`、`tile`、`compile`、`amp`、`quantize` 分支转换或报错；fraction 列表/计数特例。

### `get_save_dir`（第 547～580 行）

显式 `save_dir` 或 `increment_path(project/name)`，测试 worker 隔离。

### `_handle_deprecation`（第 583～651 行）

`end2end`→`nms`；`int8`/`half`→`quantize`；`intel:` device 映射；旧 show/hide/line 键；删除废弃键。

### `check_dict_alignment`（第 654～701 行）

先 deprecation；未知键 `get_close_matches` 提示并 `SyntaxError`。

### `merge_equals_args`（第 704～758 行）

合并 `arg = val` 分词与方括号拼接；未闭合括号报错。

### `handle_yolo_login` / `handle_yolo_settings`（第 761～835 行）

Platform API 登录/登出；settings reset 或键值更新。

### `handle_yolo_solutions`（第 838～949 行）

解析 overrides；`inference` 调 streamlit；其它解方案类，`VideoCapture` 写 `runs/solutions/exp/*.avi`；analytics 传帧号。

### `parse_key_value_pair` / `smart_value`（第 952～1033 行）

`=` 拆分；`name`/`project` 不 smart；否则 literal_eval 或 `module.CONST`。

### `entrypoint`（第 1036～1212 行）

无参打帮助；解析 special 命令；merge overrides；缺 mode 默认 predict；task 校正含 track→detect+mode track；按 stem 选 YOLO/SAM/RTDETR 等；缺 source/data/format 填默认；`getattr(model, mode)(**overrides)`。

### `copy_default_cfg`（第 1216～1238 行）

复制 default.yaml 为 `_copy.yaml` 到 cwd。

### `__main__`（第 1241～1243 行）

调用 `entrypoint()`。