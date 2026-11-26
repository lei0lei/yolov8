# `ultralytics/models/yolo/depth/predict.py`

`DepthPredictor` 继承 `BasePredictor`，`task=depth`，`scale_fill=True`；后处理输出逐像素深度 `Results.depth`，无 NMS。

## 逐段说明

### 类属性 `scale_fill`（28）

深度任务 letterbox 采用填满策略（与基类 predict 配置一致）。

### `__init__`（30–35）

`super` 后 `self.args.task = "depth"`。

### `postprocess`（37–56）

- 若 `preds` 为 tuple/list 取 `[0]` 为深度图 `(B,1,H,W)` 或 `(B,H,W)`。
- `ndim==3` 时 `unsqueeze(1)`。
- `F.interpolate` 双线性对齐到 `img` 空间尺寸（与 loss/导出一致，`align_corners=True`）。
- 非 list 的 `orig_imgs` 转 numpy BGR。
- 逐样本：深度上采样到 `orig_img` 分辨率，构造 `Results(..., depth=depth[0,0])`。