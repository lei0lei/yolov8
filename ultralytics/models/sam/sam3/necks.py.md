# `ultralytics/models/sam/sam3/necks.py`

`Sam3DualViTDetNeck` 实现 ViTDet 式 SimpleFPN：主干末层特征经多尺度转置卷积/池化与 1×1、3×3 卷积得到多 level 特征，并加位置编码。可选 `deepcopy` 第二套 conv 为 SAM2  neck 输出独立权重。`set_imgsz` 委托主干调整输入尺寸。

## 逐段说明

### `Sam3DualViTDetNeck.__init__(...)`（行 18）

- 保存 `trunk`、`position_encoding`、`scale_factors`。
- **行 46–99**：对每个 scale：
  - `4.0`：两次 ConvTranspose2d+GELU，通道逐半；
  - `2.0`：单次转置卷积；
  - `1.0`：无变换；
  - `0.5`：MaxPool2x2；
  - 否则 `NotImplementedError`；
  - 接 `Conv2d 1x1` 到 `d_model` 与 `3x3` padding=1。
- **行 101–104**：`add_sam2_neck` 时 `sam2_convs = deepcopy(convs)`。

### `forward(self, tensor_list)`（行 106）

- `xs = trunk(tensor_list)`，取 `x = xs[-1]`；
- `sam_forward_feature_levels(x, convs)` 得 SAM3 特征与 pos；
- 无 `sam2_convs` 返回后两者 None；否则再跑 SAM2 分支。

### `sam_forward_feature_levels(self, x, convs)`（行 128）

逐 conv：特征过 conv → `position_encoding(feat).to(dtype)`，收集 outs/poss。

### `set_imgsz(self, imgsz=None)`（行 139）

默认 `[1008,1008]`，调用 `self.trunk.set_imgsz`。