# `ultralytics/models/llm.py`

`LLM` 类封装 OpenAI 兼容 API（`responses` 或 `chat.completions`），懒加载同步/异步客户端。`__call__`/`async_call` 经 `_prepare` 将文本或图像归一化为 API 消息，再 `_request` 合并默认参数并调用 SDK。图像经 base64 data URL 或直传 http(s) URL。不修改 YOLO 推理链，供 Ultralytics 门面 `from ultralytics import LLM` 使用。

## 逐段说明

### 行 11–13：导入

`LoadPilAndNumpy`、`check_requirements`、`imread`；标准库 base64、Path、typing。

### `LLM.__init__(...)`（行 45）

- **行 67–68**：`api` 不在 `responses`/`chat.completions` 则 `ValueError`。
- **行 70–77**：保存 `model`、`api`、`base_url`、`prompt`、`overrides`（kwargs）；`client`/`async_client` 置 None；`_api_key` 单独保存。

### `__call__(self, source=None, image=None, **kwargs)`（行 79）

`_prepare` 后 `_call(prepared, kwargs)`。

### `_call(self, source, kwargs)`（行 95）

构建 `request`；`_get_client()`；按 `self.api` 分支调用 `responses.create` 或 `chat.completions.create`。

### `async_call` / `_async_call`（行 103–127）

与同步路径相同，使用 `_get_async_client()` 与 `await`。

### `_request(self, source, kwargs)`（行 129）

- **行 140**：`request = {model, **overrides, **kwargs}`。
- **responses**：`source` 非 None 时设 `input=source`。
- **chat.completions**：`source` 为 str 时包装单条 user 消息，否则原样作为 `messages`。
- 返回 `request` dict。

### `_prepare(self, source, image=None)`（行 148）

**无 image**：
- `source is None` → 返回构造时 `prompt`；
- list/tuple/dict → 透传原生载荷；
- str → 有 `prompt` 则 `prompt\n\n{source}`，否则 `source`；
- 其他类型视为图像：`image=source`，`prompt` 默认 `"Describe the image."`。

**有 image**：
- `prompt = source or "Describe the image."`；
- 若构造时有 `prompt`：有 `source` 则拼接，无则只用构造 `prompt`。

**行 163–182**：`_image_url(image)` 后按 API 组装 user 消息（responses 用 `input_text`/`input_image`；chat 用 `text`/`image_url` 嵌套 url）。

### `_image_url(source)` 静态方法（行 184）

- str 且 http/https/data:image → 原样返回；
- 路径或 Path → `imread`；否则 `LoadPilAndNumpy._single_check`；
- None → `ValueError`；
- `cv2.imencode(".jpg")` 失败 → `ValueError`；
- 返回 `data:image/jpeg;base64,...`。

### `_get_client` / `_get_async_client`（行 197–215）

首次调用 `check_requirements("openai>=2.0.0")`，导入 `OpenAI`/`AsyncOpenAI`；仅非 None 的 `api_key`、`base_url` 传入构造；缓存实例后返回。