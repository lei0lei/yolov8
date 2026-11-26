# `ultralytics/nn/backends/hailo.py`

本后端使用 **HailoRT**（`hailo_platform`）加载 **`.hef`**（Hailo 导出目录内递归查找）。推理管线：`HEF` → `VDevice` 配置 → `InferVStreams`，输入 **UINT8**、输出 **FLOAT32**。支持任务：`detect`、`segment`、`pose`、`obb`、`classify`、`semantic`、`depth`；其余 task 在加载时 `ValueError`。输入在 `forward` 中由 BCHW [0,1] 转为 **BHWC uint8**；多数检测类任务在主机侧用 DFL、`dist2bbox` 等与 PyTorch 头对齐的方式**解码**，或走芯片内 NMS 的 `_decode_nms`。元数据字段 `nms`、`semantic_baked`、`cal_a`/`cal_b` 等影响分支。

## 逐段说明

### 模块导入（第 1–14 行）

`math`、`ExitStack`、`Path`、`numpy`、`torch`、`LOGGER`、`BaseBackend`。

### 类 `HailoBackend`（第 17–223 行）

#### `load_model`（第 20–78 行）

1. **导入 HailoRT**：失败则 `ImportError` 与 Hailo 文档链接。
2. **找 `.hef`**：`rglob("*.hef")` 且须为文件，否则 `FileNotFoundError`。
3. **先读元数据**：`apply_metadata(read_metadata(hef_file))` 在设备配置前执行以便知 `task`。
4. **任务白名单**：`self.task` 非空且不在允许集合则 `ValueError`。
5. **`self.hef = HEF(path)`**；缓存 `input_info`（首路输入 vstream）、`output_infos`（全部输出 vstream）。
6. **`ExitStack` 持有上下文**：`VDevice()` → `ConfigureParams.create_from_hef(..., PCIe)` → `configure` 取 `network_group[0]` → `activate` → `InputVStreamParams.make(..., UINT8)`、`OutputVStreamParams.make(..., FLOAT32)` → `InferVStreams` 存 `self.model`；`stack.pop_all()` 到 `self._stack` 防栈退出即销毁。
7. **`self._anchors = None`**；若 task 为 `segment`/`pose`/`obb` 则实例化 `DFL` 为 `self._dfl`。
8. **`self.end2end`**：`self.end2end or metadata.get("nms", False)`（头选择或 HailoRT 内嵌 NMS）。

#### `__del__`（第 80–83 行）

若存在 `self._stack`，`close()` 释放 Hailo 设备与流。

#### `forward`（第 85–116 行）

**预处理**：`im` → `permute(0,2,3,1)` → CPU numpy ×255、`clip`、`uint8`、`ascontiguousarray`（BHWC）。

**推理**：`self.model.infer({input_info.name: im})`。

**输出列表**：按 `output_infos` 名称顺序从 `results` 取值到 `outputs`。

**按 `self.task` 分派**：

| 分支 | 行为 |
|------|------|
| `segment` | `_decode_segment(outputs)` |
| `pose` | `_decode_pose(outputs)` |
| `obb` | `_decode_obb(outputs)` |
| `classify` | `torch.from_numpy(outputs[0])` reshape 为 `(B, -1)` 概率 |
| `semantic` | 见下 |
| `depth` | `_decode_depth(outputs[0])` |
| 默认（detect 等） | `metadata["nms"]` 假则 `_decode_raw(outputs)`，真则 `_decode_nms(outputs[0])` |

**`semantic` 子分支**：`semantic_baked` 为元数据真时，直接 reshape 为 `(B,H,W)` 类图；否则 logits 为 NHWC，**`permute(0,3,1,2)`** 交 predictor 做上采样与 letterbox 还原。

#### `_decode_nms`（第 118–137 行）

芯片 per-class NMS 输出：坐标为归一化 **yxyx**。

- 用 `input_info.shape[:2]` 得 H、W，构造 `scale = [w,h,w,h]`。
- **逐帧** `for detections in output`：逐类 `enumerate`，非空则数组化，`boxes = [:,[1,0,3,2]] * scale`（yxyx→xyxy 像素），`classes` 填 class_id，与 score 列拼成 6 列；无检测则空 `(0,6)`。
- 每帧按 score 降序取前 **300**。
- 对齐 batch：各帧长度不一，建 `(n_frames, max_count, 6)` 零数组，拷贝各帧有效行。

#### `_decode_boxes`（第 139–150 行）

**reg_maps** 为各尺度回归特征（torch）；可选 **angle** 做 OBB。

- 首次：`strides = input_h / map_h`，`make_anchors(reg_maps, strides)` 缓存 `self._anchors`。
- `dist = DFL(cat flatten reg_maps))` 转置维度。
- **有 angle**：`dist2rbox(dist, angle, anchors) * stride`。
- **无 angle**：`dist2bbox(dist, anchors, xywh=True) * stride`。
- 返回 `(B, A, 4)` xywh（或旋转框）。

#### `_decode_segment`（第 152–162 行）

- **proto**：`outputs[-1]` NHWC → `permute` NCHW。
- `k = len(outputs)-1`；`outputs[0:k:3]` reg、`1:k:3` cls、`2:k:3` coeff（各尺度三件套）。
- `_decode_boxes(reg_maps)`；cls/cof flatten 拼接，sigmoid 已在导出烘焙。
- 返回 `[cat(boxes,cls,cof).transpose(1,2), proto]` 供 predictor NMS。

#### `_decode_pose`（第 164–178 行）

- 全部输出 `permute` NCHW；`0::3` reg、`1::3` cls、`2::3` kpt。
- boxes 同 segment；关键点 raw flatten 后按 `kpt_shape` reshape。
- **xy**：`(raw*2 + (anchor-0.5))*stride`；**ndim==3** 时 visibility 维 `sigmoid`，否则仅 xy。
- 拼接 boxes、cls、kpts flat → `transpose(1,2)` 密集输出。

#### `_decode_obb`（第 180–188 行）

- reg/cls/angle 三分组；angle sigmoid 后 `(sigmoid-0.25)*pi`。
- `_decode_boxes(reg, angle)` 得旋转 xywh；拼 cls 与 angle。

#### `_decode_raw`（第 190–212 行）

分支前半/后半为 box_maps 与 cls_maps（`split = len//2`），NHWC→NCHW。

- 锚点同 `_decode_boxes` 逻辑；**boxes** 直接 cat flatten（非 DFL 路径 YOLO26）。
- `dist2bbox(..., xywh=not self.end2end) * stride`。
- **scores**：cls flatten + **sigmoid**。
- **`not end2end`**：`cat(boxes, scores).transpose(1,2).numpy()`。
- **`end2end`**：每 anchor 类分数 topk 300，gather boxes/scores，再全局 topk 300，拼 box、score、class index → numpy。

#### `_decode_depth`（第 214–223 行）

- logit NHWC → NCHW；`clamp(-4,5).exp()`。
- 元数据 **`cal_a`**、**`cal_b`**：`depth.pow(cal_a) * exp(cal_b)` 得度量深度图（H/4 分辨率，与 `DepthPredictor` 后续 resize 一致）。