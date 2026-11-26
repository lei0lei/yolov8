# `ultralytics/utils/git.py`

`GitRepo` 在不调用 `git` 可执行文件的前提下，从磁盘上的 `.git` 元数据解析仓库根、当前分支、HEAD 提交 SHA、提交主题与 `origin` 远程 URL。支持普通仓库、gitfile 指针与工作树 `commondir`；提交消息从 zlib 压缩的 loose object 读取。构造时快照 HEAD，故训练中途 checkout 不会改变已写入 checkpoint 的版本信息。`is_repo`、`branch`、`message`、`origin` 为 property 或 cached_property。

## 逐段说明

### 类 `GitRepo`（第 10～146 行）

#### `__init__`（第 42～54 行）

- **第 49 行**：`self.root = _find_root(path or 本模块路径)`。
- **第 50～51 行**：有 root 则 `_gitdir`、`_refdir`；读 `HEAD` 文件到 `self.head`。
- **第 53 行**：`head` 以 `ref: ` 开头则 `self.ref` 为引用名，否则 detached SHA 字符串。
- **第 54 行**：有 ref 则 `_ref_commit`，否则 `commit = head`。

#### `_find_root`（第 56～59 行）

- 从路径向上遍历，首个存在 `.git` 的目录为 root，否则 `None`。

#### `_gitdir`（第 61～71 行）

- **第 65～66 行**：`.git` 为目录则直接返回。
- **第 67～70 行**：为 gitfile 且内容 `gitdir:` 则解析相对路径 resolve。
- **第 71 行**：否则 `None`。

#### `_refdir`（第 73～80 行）

- **第 77～79 行**：存在 `commondir` 文件则读路径，相对路径相对 gitdir resolve。
- **第 80 行**：默认 `refdir = gitdir`。

#### `_read`（第 82～85 行）

- 路径存在则 `read_text(errors="ignore").strip()`，否则 `None`。

#### `_ref_commit`（第 87～101 行）

- **第 89～91 行**：读 loose ref 文件得 SHA。
- **第 92～100 行**：否则扫 `packed-refs`，跳过注释与 peeled 行，匹配 ref 名得 SHA。
- **第 101 行**：未找到返回 `None`。

#### `_commit_subject`（第 103～115 行）

- **第 105～107 行**：对象路径 `objects/{前2}/{剩余}`，不存在返回 `None`。
- **第 108 行**：zlib 解压。
- **第 109～110 行**：无 `\0` 分隔则无效。
- **第 111～114 行**：类型须为 `commit `，body 双换行后第一行为 subject。
- **第 115 行**：空 subject 返回 `None`。

#### `is_repo`（第 117～120 行）

- `gitdir is not None`。

#### `branch`（第 122～125 行，cached_property）

- ref 以 `refs/heads/` 开头则返回去掉前缀的分支名，否则返回 ref 本身（detached 等）。

#### `message`（第 127～130 行）

- 有 `commit` 则 `_commit_subject(commit)`。

#### `origin`（第 132～146 行）

- **第 135～136 行**：非仓库返回 `None`。
- **第 137～145 行**：逐行解析 `config`：`[remote "origin"]` 段内 `url =` 取值。
- 未配置则隐式 `None`。