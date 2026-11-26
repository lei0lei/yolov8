# `ultralytics/utils/__init__.py`

本文件是 `ultralytics.utils` 包的根模块：在 import 时设置全局常量、环境变量、日志、默认配置与用户设置，并 re-export 补丁与 tqdm。除工具函数外还提供自研 YAML 读写器、可迭代配置命名空间、JSON 持久化设置、Sentry 条件初始化，以及大量 `is_*` 运行环境探测。文件末尾实例化 `SETTINGS`、路径常量，应用 `torch.save` 与 Windows 下 cv2 路径补丁。几乎所有子模块都依赖此处 `LOGGER`、`ROOT`、`DEFAULT_CFG` 与 `USER_CONFIG_DIR`。

## 逐段说明

### 导入与 `env_bool`（第 3～61 行）

懒处理 torch 导入失败时 `import_execstack`。`env_bool`：`getenv` 为 None 返回 default，否则 strip lower 后在真值集合内判 bool。

### 分布式与全局常量（第 64～205 行）

`WORLD_SIZE>1` 时解析 `RANK`/`LOCAL_RANK` 否则 -1。定义 `ROOT/ASSETS/DEFAULT_CFG_PATH/NUM_THREADS/AUTOINSTALL/VERBOSE/SAFE_LOAD`、平台与导出芯片表、`HELP_MSG`。模块级设置 torch/numpy 打印、OpenCV 单线程、TF/torch 日志级别、warnings 过滤、`FLOAT_OR_INT`/`STR_OR_PATH` 类型元组。

### 类 `DataExportMixin`（第 207～283 行）

`to_df` 懒 import polars 调 `summary()`。`to_csv` 先 DataFrame，`write_csv` 失败则复杂类型转 str 列再写。`to_json` 经 DataFrame `write_json`。

### 类 `SimpleClass`（第 286～332 行）

`__str__` 列举非 callable 非 `_` 属性，嵌套 SimpleClass 只显示类型名。`__repr__` 同 str。`__getattr__` 抛带 doc 的 AttributeError。

### 类 `IterableSimpleNamespace`（第 335～391 行）

`__iter__` 迭代 `vars` 项；`__str__` 每行 `k=v`。缺失键错误信息指向更新 default.yaml。`get` 等同 `getattr` default。

### `plt_settings`（第 394～465 行）

装饰器工厂，默认 rc `font.size=11`。内层 wrapper：首次注册 `USER_CONFIG_DIR` 下 ttf 到 matplotlib；无 `font.sans-serif` 时前置 Arial Unicode。切换 backend 前后 `close all` 与还原；`rc_context` 执行被装饰函数。

### `set_logging` / `LOGGER` / `emojis`（第 468～547 行）

rank 非 0/-1 或非 verbose 时 ERROR，否则 INFO。`PrefixFormatter` 给 WARNING/ERROR 加前缀与 emoji（Windows 去 emoji）。Windows 尝试 stdout UTF-8。移除同名 handler 后挂 StreamHandler，`propagate=False`。全局 `LOGGER` 用 `VERBOSE`；sentry_sdk 级别压低。`emojis` Windows 剥离指定 emoji。

### 类 `ThreadingLocked`（第 550～580 行）

实例持 Lock；装饰后 `with self.lock` 调用原函数。

### 类 `YAML`——类属性与 IO（第 583～740 行）

内置 `_CONST` 标量映射、转义表、正则。`save`：非可序列化顶层值转 str，`_dump` 写 utf-8。`load` 断言后缀，parse 失败包装 ValueError；非 dict 拒绝；可选 `yaml_file` 键。`print` load 或直接用 dict 后 info 输出 `_dump`。`_dump` 块样式 dict/list，空集合内联。`_repr` 处理 None/bool/int/float/nan inf、YAML1.1 敏感字符串 json 引号。`_scalar` 解析 bool/int/float/ str。`_unquote` 单双引号折行与转义。

### 类 `YAML.parse` 与 `_peek`（第 766～803 行）

`parse` 循环剥 `%` 指令行与 `---` 文档头；`_node(-1)` 得根；尾部仍有内容 `_error`。`_peek` 递增 `i`，空行与 `#` 注释跳过；非空行计算空格缩进，遇 tab `_error`。

### 类 `YAML._key` / `_node` / `_map`（第 806～839 行）

引号键用 `_QUOTED`+冒号后 rest；plain 键找 `: ` 或尾冒号，拒绝注释嵌入键。`-` 开序列、可解析键则 `_map`，否则 `_value`。映射循环同缩进读 `key: value` 写入 dict。

### 类 `YAML._seq`（第 841～861 行）

`-` 同行有 rest 则 `_value` 或重排行为嵌套节点；空项 peek 下一级为块则 `_node` 否则 None。缩进大于 n 且非合法 `-` 行则 `_error`。

### 类 `YAML._value`（第 863～895 行）

空 rest 且下一行更深：嵌套块或同级 `-` 序列；`[{` 或引号开头走 `_flow`/`_unquote` 多行拼接。`|>` 走 `_block_scalar`。`&*!` 报错不支持。否则剥行内 `#` 后可能拼接多行 plain；含疑似 mapping 内嵌则 `_error`；最后 `_scalar`。

### 类 `YAML._block_scalar`（第 897～922 行）

头匹配 `|>` 与可选数字 chomp 符。累积行直至缩进≤父级；推断或固定内容缩进列。`>` 模式折叠非空行间换行为空格。`-` 或文件末则不追加尾换行；`+` 保留尾随空行数。

### 类 `YAML._flow` / `_flow_end`（第 924～970 行）

`_flow_end` 追踪 `[]{}` 深度至闭合。`_flow` token 递归：引号/ plain/ 嵌套集合；冒号后是 `}`/`,` 则值 None；缺逗号非闭合 `_error`。

### 默认配置（第 973～977 行）

`YAML.load(DEFAULT_CFG_PATH)` 得 `DEFAULT_CFG_DICT/KEYS` 与 `IterableSimpleNamespace` 实例 `DEFAULT_CFG`；`PREDICT_IOU=0.5`。

### 设备与 OS 探测函数（第 980～1197 行）

`read_device_model`：`platform.release().lower()`。`cpu_threads`：Linux 下 affinity 与 cgroup cpu.max 限制，上限 8。`is_ubuntu`/`is_debian` 读 os-release。`is_colab`/`is_kaggle`/`is_jupyter`/`is_runpod`/`is_docker` 环境变量或 `/.dockerenv`。`is_raspberrypi` 查 `DEVICE_MODEL`。`is_jetson` optional jetpack 读 tegra release。`is_dgx` 读 dgx-release。`is_online` 尊重 `YOLO_OFFLINE`，DNS 解析 one.one.one.one/dns.google。`is_pip_package` importlib find_spec origin。`is_dir_writeable` access W_OK。`is_pytest_running` 环境或 argv。`is_github_action_running` 三键 GHA 环境变量。

### `get_default_args` / `get_ubuntu_version`（第 1209～1233 行）

inspect 默认参数 dict；Ubuntu 时正则 VERSION_ID。

### `get_user_config_dir`（第 1236～1274 行）

`YOLO_CONFIG_DIR` 优先；否则 XDG/Windows Roaming/macOS Application Support；mkdir 失败则 tempfile 或 mkdtemp 私有目录并 warning。

### 二次常量与 `colorstr`（第 1277～1346 行）

`DEVICE_MODEL` 后批量 `IS_*`、`GIT`、`USER_CONFIG_DIR`、`SETTINGS_FILE`。`colorstr` 解析 ANSI 颜色表包装字符串。

### `remove_colorstr` / `TryExcept` / `Retry`（第 1349～1448 行）

剥离 ANSI 转义。`TryExcept` 上下文/装饰器：verbose 时 warning，`__exit__` 吞异常返回 True。`Retry` 装饰：`times` 次尝试，失败指数退避 sleep，`verbose` 打 retry 日志。

### `threaded`（第 1451～1485 行）

`kwargs.pop("threaded",True)` 为真则非 daemon `Thread` 启动并返回线程，否则同步执行返回结果。

### `set_sentry`（第 1488～1560 行）

需 settings sync、rank、yolo CLI、在线 pip 非 git 等全满足；`before_send` 过滤 KeyboardInterrupt/OOM 等，附 tags。init dsn、关局部变量、sample rate、uuid 用户。

### 类 `JSONDict`（第 1563～1653 行）

`__init__` 设 path、Lock、`_load` json。`_save` mkdir 写 indent=2，Path 用 `_json_default` 转 str。`__setitem__`/`__delitem__`/`update`/`clear` 持锁后 `_save`。

### 类 `SettingsManager`（第 1656～1784 行）

`__init__` 默认 dict 含目录、uuid、sync、api_key、各集成开关；`torch_distributed_zero_first` 下 super 初始化；无文件则 reset；`_validate_settings` 键类型版本对齐，非法 api_key 清空，datasets_dir≠runs_dir 警告。`__setitem__` 转 `update`。`update` 校验键在 defaults 且类型一致。`reset` 写 defaults。

### 杂项与包尾（第 1787～1842 行）

`deprecation_warn` LOGGER warning。`clean_url` Path+unquote 去 query。`url2file` basename。`vscode_msg` 检查扩展目录与 obsolete 列表。`SETTINGS` 实例化→`DATASETS_DIR`/`WEIGHTS_DIR`/`RUNS_DIR`/`ENVIRONMENT`/`TESTS_RUNNING`→`set_sentry()`。`torch.save=torch_save`；Windows 替换 cv2 imread/imwrite/imshow。