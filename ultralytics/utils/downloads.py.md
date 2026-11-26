# `ultralytics/utils/downloads.py`

本模块是 Ultralytics 资源获取的核心：维护 GitHub releases 上已知权重名集合、带断点续传与 Content-Length 校验的 `safe_download`、zip/tar 安全解压、磁盘空间检查、Release API 查询与 `attempt_download_asset` 本地/URL/GitHub 统一入口，以及多线程 `download` 批处理。支持 curl 回退、传输编码解压与 zip-slip 路径校验。

## 逐段说明

### 常量 `GITHUB_ASSETS_*`（第 20～62 行）

- `GITHUB_ASSETS_REPO`、`GITHUB_ASSETS_NAMES` 冻结合法文件名集合；`GITHUB_ASSETS_STEMS` 为无扩展名 stem 集。

### `is_url`（第 65～90 行）

- `urlparse` 需 scheme 与 netloc；`check=True` 时 `requests.head` 判可达。

### `delete_dsstore`（第 93～115 行）

- rglob 匹配 `.DS_Store`/`__MACOSX`，目录 `rmtree` 否则 `unlink`。

### `zip_directory`（第 118～159 行）

- 先 `delete_dsstore`；非目录抛 `FileNotFoundError`。
- 收集需打包文件；`ZipFile` DEFLATED/STORED；TQDM 写入相对路径。

### `unzip_file`（第 162～232 行）

- 存在且为 zip 否则 `BadZipFile`；默认解压父目录。
- 根据顶层目录数量决定解压到子目录还是 `{stem}/`。
- 已存在且非空且非 `exist_ok` 则 warning 跳过。
- 逐成员 resolve 路径，拒绝 `..` 与越界；TQDM extract。

### `check_disk_space`（第 235～275 行）

- `shutil.disk_usage`；`file_bytes*sf < free` 或 total=0 视为充足。
- 不足时 `fmt_bytes` 格式化消息；`hard` 抛 `MemoryError` 否则 warning 返回 False。

### `safe_download`（第 278～496 行）— 本地与 URL 准备

- **第 325～326 行**：无 `://` 且本地存在则 `f=Path(url)` 走尾部解压逻辑。
- **第 328～338 行**：远程则 requests/curl 分支；目标路径、`.part` 临时名、mkdir。

### `safe_download` — 重试循环体内

- **第 346 行**：`resume` 为已下载 part 大小。
- **第 347～364 行**：curl 分支 stall 限制、`-f` 防错页；解析最终响应 Content-Encoding。
- **第 365～394 行**：requests 流式写；416 删 part；非 206 重置 resume；Range 续传；大文件 `check_disk_space`；TQDM 进度。
- **第 396～434 行**：大小与 Content-Length 比对；identity 以外编码则 zlib/gzip 逐块解压到 `.decoded` 再 replace（保留双重 gzip 的 tar.gz 情形）；大于 `min_bytes` 则 `replace` 到 target 并 break。
- **第 435～456 行**：`MemoryError` 直接 raise；其它错误离线首轮删 part 抛 ConnectionError；4xx 快失败；指数退避 sleep。

### `safe_download` — 解压与返回

- **第 458～495 行**：后缀匹配 archive 集合则 zip `unzip_file` 或 tar 安全提取（路径校验、保留 mtime）；`delete` 删归档；返回目录或文件 Path。
- **第 496 行**：非解压则返回 `f`。

### `get_github_assets`（第 499～540 行）

- version 非 latest 则 `tags/{version}` API URL；可重试一次。
- 200 或 rate limit reason 退出循环；否则 warning 空 tag/assets。
- 返回 `tag_name` 与 assets 名列表。

### `attempt_download_asset`（第 543～594 行）

- `check_yolov5u_filename` 规范化；SETTINGS weights_dir 或 cwd 存在即返回。
- HTTP URL：`url2file` 本地命中则 info；否则 `safe_download`。
- 官方 assets 名：`releases/download/{release}/{name}`。
- 否则 `get_github_assets` 查 tag，失败再 latest，命中则 download。

### `download`（第 597～648 行）

- 确保 dir；urls 列表化。
- `threads>1` 用 ThreadPool `map` safe_download；否则顺序调用。