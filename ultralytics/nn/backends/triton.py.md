# `ultralytics/nn/backends/triton.py`

本后端不加载本地权重，而是连接 **NVIDIA Triton Inference Server** 上已部署的模型。`weight` 为 **`http://` 或 `grpc://` URL**（如 `http://localhost:8000/model_name`）。依赖 **`tritonclient[http,grpc]`**；封装类为 `ultralytics.utils.triton.TritonRemoteModel`。若远端模型对象带 **`metadata`** 属性，则 `apply_metadata` 同步到本地后端。输入 **BCHW [0,1]** numpy；返回服务器输出的 numpy 列表。

## 逐段说明

### 模块导入（第 1–11 行）

`Path`、`torch`、`check_requirements`、`BaseBackend`。

### 类 `TritonBackend`（第 14–45 行）

#### `load_model`（第 21–34 行）

1. `check_requirements("tritonclient[http,grpc]")`。
2. `self.model = TritonRemoteModel(weight)`——在工具类内完成协议选择、模型名解析与客户端初始化（本文件无进一步分支）。
3. **`hasattr(self.model, "metadata")` 真**：`apply_metadata(self.model.metadata)`；否则不读本地侧车元数据文件。

#### `forward`（第 36–45 行）

1. `return self.model(im.cpu().numpy())`——将 BCHW float batch 交给远程封装；`TritonRemoteModel` 负责按服务端 tensor 契约投递并取回多输出。本层无 permute、量化或输出重排。

输出张量如何拆分为各检测头/任务输出，由 Triton 模型配置与 Ultralytics 远端元数据约定，在 `TritonRemoteModel` 与 predictor 中完成，不在此文件内。