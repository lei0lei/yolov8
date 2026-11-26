# `ultralytics/models/sam/sam3/vl_combiner.py`

`SAM3VLBackbone` 组合 `Sam3DualViTDetNeck` 与语言编码器，训练可选整网 activation checkpoint 与 `torch.compile` 视觉分支。`forward` 合并图像与文本字典；`scalp` 丢弃最低分辨率 FPN level。SAM2 特征可单独经 `forward_image_sam2` 取得。

## 逐段说明

### `__init__(...)`（行 25）

- 视觉：`torch.compile(visual)` 或原模块；
- 保存 text、`scalp`、整网 ckpt 旗标。

### `forward(samples, captions, input_boxes=None, additional_text=None)`（行 52）

`forward_image` 与 `forward_text` 结果 `update` 合并返回。

### `forward_image(samples)`（行 81）

- 调 `vision_backbone.forward` 得四元组；
- `scalp>0` 截断 sam3/sam2 列表尾部；
- 若有 SAM2：组 `sam2_backbone_out` 字典（最后 level 为 `vision_features`）；
- SAM3 同样以最后一 level 为 `vision_features`，附 `backbone_fpn`、`vision_pos_enc`。

### `forward_image_sam2(samples)`（行 115）

仅 `trunk`+`sam2_convs` 路径，scalp 后返回 SAM2 风格字典。

### `forward_text(captions, input_boxes=None, additional_text=None)`（行 136）

- `text_to_encode = captions`；有 `additional_text` 则拼接待一并编码；
- `sdpa_kernel` 包裹下调用 `language_backbone`；
- 若有附加文本：切片尾部写入 `additional_text_features/mask`；
- 截回 captions 长度得 `language_features`、`language_mask`、`language_embeds`。

### `set_imgsz(self, imgsz=None)`（行 163）

默认 1008，委托 `vision_backbone.set_imgsz`。