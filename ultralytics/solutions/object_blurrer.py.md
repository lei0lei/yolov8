# `ultralytics/solutions/object_blurrer.py`

对检测框区域应用均值模糊以保护隐私：`blur_ratio` 转为内核边长；在 `im0` 上就地替换 ROI，再绘制框标签。

## 逐段说明

### `__init__`（第 34～46 行）

`blur_ratio<0.1` 警告并回退 0.5；`self.blur_ratio = int(ratio*100)`。

### `process`（第 48～90 行）

`extract_tracks`；对每个框 `get_enclosing_box` 裁剪并夹紧到图像；空 ROI 跳过；`cv2.blur` 写回；`box_label`；返回 `total_tracks=len(track_ids)`。