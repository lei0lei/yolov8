# `ultralytics/utils/cpu.py`

本模块提供跨平台 CPU 型号字符串的获取与规范化，类 `CPUInfo` 以静态方法为主，无需实例化也可通过 `CPUInfo.name()` 使用。macOS 经 libc `sysctlbyname` 读品牌串；Linux 流式扫描 `/proc/cpuinfo` 首条 `model name`；Windows 读注册表 `ProcessorNameString`。均失败时回退 `platform.processor()` / `uname` / `machine`。`_clean` 去掉商标符号并尝试将 Intel/AMD 常见格式压缩为「型号 + 频率」短串。`__str__` 等同 `name()`。

## 逐段说明

### 类 `CPUInfo`（第 12～101 行）

#### `_sysctl`（第 31～49 行）

- **第 34～42 行**：通过 `ctypes.CDLL(None)` 绑定 `sysctlbyname` 参数与返回类型。
- **第 43～45 行**：编码 key，先以 `buf=None` 探测所需缓冲区长度 `n`；失败或 `n==0` 返回空串。
- **第 46～48 行**：分配 buffer 再次调用；失败返回空串。
- **第 49 行**：解码为字符串并 strip。

#### `name`（第 51～86 行）

- **第 54 行**：外层 `try` 包裹各平台分支。
- **第 55～59 行**：`darwin` 分支调用 `_sysctl("machdep.cpu.brand_string")`，非空则 `_clean` 返回。
- **第 60～68 行**：`linux` 打开 `/proc/cpuinfo`，首行含 `model name` 则取冒号后内容 `_clean` 返回。
- **第 69～79 行**：`win` 分支用 `winreg` 读 `CentralProcessor\0` 的 `ProcessorNameString`；异常则 pass 走通用回退。
- **第 81～82 行**：通用回退链 `platform.processor()` → `uname.processor` → `machine`，再 `_clean`，默认 `"Unknown CPU"`。
- **第 83～86 行**：任意异常时用 `processor`/`machine` 拼串，仍保证返回字符串。

#### `_clean`（第 88～97 行）

- **第 91 行**：折叠空白为单空格并 strip。
- **第 92 行**：移除 `(TM)`/`(R)` 等标记。
- **第 93～94 行**：正则匹配 Intel `i*` CPU @ GHz，返回拼接组。
- **第 95～96 行**：同理匹配 AMD Ryzen 模式。
- **第 97 行**：无匹配则返回整句清理后字符串。

#### `__str__`（第 99～101 行）

- 返回 `self.name()`。

### 模块 `__main__`（第 104～105 行）

- 打印 `CPUInfo.name()` 供本地测试。