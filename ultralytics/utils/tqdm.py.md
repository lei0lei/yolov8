# `ultralytics/utils/tqdm.py`

`TQDM` 是零外部依赖的进度条实现，用于替代第三方 tqdm，兼容训练日志、W&B 与 `ConsoleLogger.progress` 钩子。支持迭代器协议、上下文管理器、字节单位缩放、终端宽度自适应截断与 GitHub Actions 等非交互环境下的降频/禁用回车刷新。`is_noninteractive_console` 检测 `GITHUB_ACTIONS` 环境。

## 逐段说明

### `is_noninteractive_console`（第 16～19 行）

- `@lru_cache(maxsize=1)`：环境含 `GITHUB_ACTIONS` 返回 True。

### 类 `TQDM`（第 22～414 行）

#### 类常量（第 78～82 行）

- 速率计算最小间隔、EMA 平滑系数、平滑上限、非交互最小刷新间隔 60s。

#### `__init__`（第 84～159 行）

- **第 119～125 行**：`disable is None` 时按 `VERBOSE` 与 LOGGER 有效级别 > INFO（20）决定是否禁用。
- **第 127～129 行**：`total` 默认 `len(iterable)`，显式 0 不覆盖为 0（用 `or None`）。
- **第 135～136 行**：非交互时 `mininterval` 至少 60s。
- **第 140～142 行**：`bar_format` 仅 API 兼容；`file` 默认 stdout。
- **第 145～150 行**：计数 `n=initial`、打印状态、起止时间、速率、`closed=False`。
- **第 151～156 行**：`is_bytes` 与 `scales` 单位表（字节或通用 it）。
- **第 158～159 行**：未禁用、有 total、非非交互时初次 `_display()`。

#### `_format_rate`（第 161～174 行）

- rate≤0 返回空串。
- inv_rate>1 用 s/it（字节用 s/B）否则 it/s，大速率走 scales 分档。

#### `_format_num`（第 176～185 行）

- 非字节或不缩放则 `str(num)`。
- 否则 K/M/G/T 档除 `unit_divisor` 格式化，末尾 PB。

#### `_format_time`（第 187～194 行）

- &lt;59.95s 一位小数秒；否则分:秒或时:分:秒。

#### `_generate_bar`（第 196～206 行）

- 无 total：闭合用 `━` 否则 `─` 固定宽度。
- 有 total：按比例填充 `━`/`─`，过半格用 `╸`。

#### `_fit`（第 208～223 行）

- 跳过 ANSI 转义宽度；CJK 计双宽；超宽截断加 `…\033[0m`。

#### `_should_update`（第 225～229 行）

- 非交互永不刷新；否则完成或 `dt >= mininterval`。

#### `_display`（第 231～314 行）— 速率与字段

- **第 233～234 行**：禁用或已关闭且非 final 则 return。
- **第 240～241 行**：非 final 且不应更新则 return。
- **第 244～251 行**：间隔够长则用 dn/dt 更新速率并对中等速率 EMA；否则沿用 `last_rate`。
- **第 254～257 行**：已达 total 用总 elapsed 平均速率。
- **第 259～261 行**：更新 `last_print_n/t`。
- **第 265～268 行**：未完成时估剩余时间字符串。
- **第 271～279 行**：百分比你 floor；字节完成时可能去掉重复后缀。
- **第 281～282 行**：elapsed 与 rate 字符串。
- **第 284～294 行**：拼 bar 与 fields（有/无 total，字节完成简化行）。

#### `_display` — 写出

- **第 297～305 行**：TTY 且非 Jupyter 则按终端列宽 `_fit` 描述与整行。
- **第 307～312 行**：非交互无前导 `\r\033[K`；若 `file.progress` 存在则回调 `(id(self), frame)` 否则 write+flush。
- **第 313～314 行**：写失败静默。

#### `update`（第 316～320 行）

- 未禁用未关闭则 `n += n`（默认 1）并 `_display()`。

#### `set_description` / `set_postfix`（第 322～333 行）

- 更新 `desc` 并刷新；postfix 拼到 `base_desc | k=v,...`。

#### `close`（第 335～363 行）

- 已关闭 return；置 closed。
- 有 total 且 n≥total 对齐 n；final `_display`。
- 通知 `file.progress(id, "")`；`leave` 写换行否则清行。
- flush 异常忽略。

#### 上下文与迭代（第 365～384 行）

- `__enter__` 返回 self；`__exit__` 调 close。
- `__iter__`：无 iterable 抛 TypeError；finally close。

#### `__del__`（第 385～390 行）

- try close。

#### `refresh` / `clear` / `write`（第 392～414 行）

- refresh 调 `_display`；clear 写 `\r\033[K`；静态 write 写字符串+换行。

### 模块 `__main__`（第 417～472 行）

- 六种用法演示：已知 total 迭代、手动更新、未知 total、context manager、流式迭代、文件扫描。