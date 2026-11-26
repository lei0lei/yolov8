# `ultralytics/utils/uploads.py`

本模块提供带重试与可选进度条的 HTTP 文件上传，模式与 `downloads.safe_download` 对称。`safe_upload` 对签名 URL 等场景使用 `requests.put` 流式上传，正文经 `_ProgressReader` 包装以便 `TQDM` 按字节更新。客户端 4xx（除 408/429）快速失败不重试；其余错误指数退避重试。成功时打 info 日志并返回 `True`。

## 逐段说明

### 类 `_ProgressReader`（第 13～34 行）

#### `__init__`（第 16～19 行）

- 以二进制打开 `file_path`；保存 `pbar`；`_size = os.path.getsize`。

#### `read`（第 21～26 行）

- `file.read(size)` 后若有数据且存在 pbar，则 `pbar.update(len(data))`。
- 返回读到的 `data`（供 requests 消费）。

#### `__len__`（第 28～30 行）

- 返回 `_size`，供 requests 设置 Content-Length。

#### `close`（第 32～34 行）

- 关闭底层文件对象。

### `safe_upload`（第 37～118 行）

- **第 65 行**：懒 import `requests`。
- **第 67～69 行**：`Path` 化并存在性检查，不存在抛 `FileNotFoundError`。
- **第 71～72 行**：`file_size` 与进度条描述 `Uploading {name}`。
- **第 75～77 行**：默认 `Content-Type: application/octet-stream`，合并用户 headers。
- **第 79 行**：`last_error` 初始化。
- **第 80～96 行**：`for attempt in range(retry + 1)`：可选 TQDM 与 `_ProgressReader`；`put` 后 `raise_for_status`；关闭 reader/pbar；成功 info 并 `return True`。
- **第 98～102 行**：`HTTPError` 且 400≤status<500 且非 408/429 则 warning 并 `return False`；否则记 `last_error`。
- **第 104～105 行**：其它异常记 `str(e)`。
- **第 106～110 行**：`finally` 确保 reader/pbar 关闭。
- **第 112～115 行**：未达最大次数则 `2**(attempt+1)` 秒等待后重试。
- **第 117～118 行**：耗尽重试 warning 并 `return False`。