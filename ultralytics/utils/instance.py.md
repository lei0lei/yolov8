# `ultralytics/utils/instance.py`

本模块定义检测标注的核心数据结构：`Bboxes` 管理 xyxy/xywh/ltwh 格式 bbox 数组及坐标变换；`Instances` 在 bbox 之上统一持有可选 segments 与 COCO 式 keypoints，并提供缩放、归一化、翻转、裁剪、clip 与拼接等几何运算。大量操作委托 `ops` 中的坐标转换与 `segment2box`、`resample_segments`。面向数据增强与标签对齐，不负责归一化坐标系之外的业务逻辑。

## 逐段说明

### `_ntuple`（第 15～22 行）

- 返回闭包 `parse(x)`：若 `x` 非可迭代则 `tuple(repeat(x, n))`，否则原样返回。

### `to_2tuple` / `to_4tuple`（第 25～26 行）

- 分别为 `_ntuple(2)` 与 `_ntuple(4)` 工厂。

### 类 `Bboxes`（第 36～181 行）

#### `__init__`（第 64～76 行）

- 断言 `format in _formats`；1D 数组扩为 `(1,4)`；断言 2D 且第二维为 4；保存 `bboxes`、`format`。

#### `convert`（第 78～94 行）

- 目标 format 非法 assert；已与目标相同则 return。
- **第 87～92 行**：按当前 format 选 `xyxy2xywh`/`xyxy2ltwh`/`xywh2xyxy` 等函数变换数组。
- **第 94 行**：更新 `self.format`。

#### `areas`（第 96～102 行）

- xyxy：`(x2-x1)*(y2-y1)` 按列差；xywh/ltwh：`w*h` 用后两列。

#### `mul`（第 104～118 行）

- 标量扩为四元 `to_4tuple`；断言长度 4；四列分别乘 scale。

#### `add`（第 120～134 行）

- 标量扩四元；四列分别加 offset。

#### `__len__`（第 136～138 行）

- 返回 `len(self.bboxes)`。

#### `concatenate`（第 140～162 行）

- 空列表返回 `cls(np.empty((0,4)))`；全为 Bboxes；单元素直接返回；否则 `np.concatenate` 轴拼接，format 取首元素。

#### `__getitem__`（第 164～181 行）

- 索引后 1D reshape 为 `(1,-1)`；断言 2D；新 `Bboxes` 保留 format。

### 类 `Instances`（第 184～553 行）

#### `__init__`（第 221～241 行）

- `self._bboxes = Bboxes(...)`；`keypoints`；`normalized`；`segments` 默认 `(0,0,2)` float32 空数组。

#### `convert_bbox`（第 243～249 行）

- 委托 `_bboxes.convert`。

#### `bbox_areas`（第 251～254 行）

- property 返回 `_bboxes.areas()`。

#### `scale`（第 256～271 行）

- bbox `mul (sw,sh,sw,sh)`；`bbox_only` 则 return。
- segments 的 x/y 乘 sw/sh；keypoints 同理。

#### `denormalize`（第 273～288 行）

- 已非 normalized 则 return；bbox/segments/keypoints 乘 w,h；`normalized=False`。

#### `normalize`（第 290～305 行）

- 已 normalized 则 return；各坐标除 w,h；`normalized=True`。

#### `add_padding`（第 307～320 行）

- 断言非 normalized；bbox/segments/keypoints x/y 加 padw/padh。

#### `__getitem__`（第 322～342 行）

- 整型 index 包成列表；segments/keypoints/bboxes 同步索引；新 `Instances` 保留 format 与 normalized。

#### `flipud`（第 344～361 行）

- xyxy：交换 y1/y3 相对 h 镜像；ltwh：改 y 与 h；xywh：改中心 y。
- segments/keypoints y 轴 `h - y`。

#### `fliplr`（第 363～380 行）

- 与 flipud 对称，沿宽度 w 镜像 x。

#### `clip`（第 382～441 行）— 预备与 OBB 分支

- **第 394～395 行**：记原 format，转 xyxy。
- **第 396～424 行**：`preserve_obb` 且有 segments：对越界实例用 `intersectConvexConvex` 与画布求交；无可见区则 bbox/segment 清零；否则更新 bbox 为可见 minmax，并按原 segment 角度用 `minAreaRect` 在局部坐标系重采样四角。
- **第 425～429 行**：否则 xyxy 与 segment 坐标 clip 到 [0,w]/[0,h]。
- **第 430～431 行**：恢复原 bbox format。
- **第 432～441 行**：keypoints 越界先 visibility 置 0 再 clip x/y。

#### `crop`（第 443～456 行）

- 窗口 `(x1,y1,x2,y2)`，记原面积；转 xyxy；`add_padding(-x1,-y1)`。
- 非 preserve_obb 且有 segments：外点用 `segment2box` 收紧 bbox 并 clip segment 到 box。
- `clip` 后返回 mask：面积>0 且剩余面积 ≥ 原面积 10%。

#### `remove_zero_area_boxes`（第 458～471 行）

- `good = bbox_areas > 0`；有零面积则同步过滤 bboxes/segments/keypoints。
- 返回 `good` 布尔数组。

#### `update`（第 473～485 行）

- 用新 bboxes 重建 `Bboxes`（保持原 format）；可选替换 segments/keypoints。

#### `__len__`（第 487～489 行）

- `len(self.bboxes)`（property 来自 `_bboxes`）。

#### `concatenate`（第 491～535 行）

- 空返回空 Instances；单元素返回自身。
- 判定是否统一 keypoints；拼接 bboxes；若 segment 点数不一致则 `resample_segments` 到 max_len 再 concat；keypoints 可选 concat。
- 新实例用首元素 format 与 normalized。

#### `bboxes` property（第 537～540 行）

- 返回 `self._bboxes.bboxes` ndarray。

#### `__repr__`（第 542～553 行）

- 遍历 `__dict__`，`_bboxes` 显示为 bboxes property；非 None 字段逐行拼进 `Instances(...)`。