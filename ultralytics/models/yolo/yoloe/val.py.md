# `ultralytics/models/yolo/yoloe/val.py`

`YOLOEDetectValidator` 在验证前为 `YOLOEModel` 注入文本或视觉 prompt 嵌入；`YOLOESegValidator` 叠加分割指标与 mask 后处理。

## 逐段说明

### `get_visual_pe`（48–96）

- 断言 unfused YOLOE。
- names 截断 `/`；`visual_pe` 零初始化；统计每类 batch 出现次数 `cls_visual_num`。
- 遍历 dataloader：`preprocess`；`model.get_visual_pe(img, visuals)`。
- 按图把同类 embedding 累加并除以类样本数。
- 有样本的类 L2 归一化；无样本维保持 0。
- 返回 `(1, num_classes, embed_dim)`。

### `get_vpe_dataloader`（98–131）

`build_yolo_dataset` val；追加 `LoadVisualPrompt`；`build_dataloader` shuffle=False。

### `get_model`（133–199）

- `super().get_model` 得基础模型。
- **trainer 路径**：设备与 names 来自训练 dataloader；`load_vp` 则关 quantize、`get_visual_pe(self.dataloader)`、`set_classes`；否则 `get_text_pe` + `set_classes`。
- **独立 val**：可选 `refer_data` 仅 VP；`load_checkpoint`；`check_det_dataset`；refer 与 eval 类名不一致 warning。
- `load_vp`：外部分 dataloader 提 VPE；否则非 LRPC 融合头用 text pe。
- 返回已 `set_classes` 的 model。

### `YOLOESegValidator`（202–203）

继承 `YOLOEDetectValidator` 与 `SegmentationValidator`（MRO 合并 VP 准备与 mask 验证）。