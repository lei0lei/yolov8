# `ultralytics/solutions/security_alarm.py`

`SecurityAlarm` 跟踪并画框；当检测数≥`records` 且尚未发信且已 `authenticate` 时发带 JPEG 附件的 SMTP 邮件，并置 `email_sent=True`。

## 逐段说明

### `__init__`（第 36～47 行）

`records` 阈值，`server`/`to_email`/`from_email` 初始空。

### `authenticate`（第 49～69 行）

Gmail SMTP 587 STARTTLS 登录，保存收发地址。

### `send_email`（第 71～113 行）

JPEG 编码帧；MIME 多部分文本+图像；`send_message`，成败打日志。

### `process`（第 115～151 行）

跟踪画框；`total_det>=records` 且条件满足则 `send_email`；返回 `email_sent`。