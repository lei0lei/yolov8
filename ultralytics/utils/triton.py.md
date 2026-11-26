# `ultralytics/utils/triton.py`

`TritonRemoteModel` 是面向 NVIDIA Triton Inference Server 的轻量客户端：根据 `scheme` 选择 HTTP 或 gRPC 的 `tritonclient`，拉取模型 config，缓存输入输出名、TRITON 类型与对应 numpy dtype，以及可选的 `metadata` 参数字符串。`__call__` 将 numpy 输入按配置转型、封装 `InferInput`、请求全部输出并转回与首个输入相同的 dtype。URL 可在未单独传 `endpoint`/`scheme` 时从完整 URL 解析。

## 逐段说明

### 类 `TritonRemoteModel`（第 11～112 行）

#### `__init__`（第 40～83 行）

- **第 51～55 行**：若 `endpoint` 与 `scheme` 皆空，则 `urlsplit` 得 scheme、netloc 为 url、path 第一段为 endpoint。
- **第 57～58 行**：保存 `self.endpoint`、`self.url`。
- **第 61～65 行**：`scheme == "http"` 时 import `tritonclient.http`，建 `InferenceServerClient`，`get_model_config(endpoint)`。
- **第 66～70 行**：否则 gRPC 客户端，`get_model_config(..., as_json=True)["config"]`。
- **第 73 行**：输出按 name 字母排序，保证 output0/1 顺序稳定。
- **第 76 行**：`type_map` 映射 FP32/FP16/UINT8 到 numpy 类型。
- **第 77～78 行**：保存客户端侧的 `InferRequestedOutput`、`InferInput` 类引用。
- **第 79～82 行**：从 config 提取 input/output 的 data_type、name 列表。
- **第 83 行**：`metadata` 从 parameters.metadata.string_value 经 `ast.literal_eval`，缺省 `"None"`。

#### `__call__`（第 85～112 行）

- **第 100～107 行**：对每个输入，dtype 与期望不符则 `astype`；建 `InferInput(name, shape, type 去 TYPE_ 前缀)`；`set_data_from_numpy`。
- **第 109 行**：为每个 output_name 建 `InferRequestedOutput`。
- **第 110 行**：`triton_client.infer(model_name=endpoint, inputs, outputs)`。
- **第 112 行**：列表推导各输出 `as_numpy` 并 astype 为 `inputs[0].dtype`。