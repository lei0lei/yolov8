# `ultralytics/models/fastsam/utils.py`

提供 FastSAM 预测后处理用的框边界微调工具。当检测框靠近原图边缘且在阈值像素内时，将对应坐标吸附到图像边界（0 或宽/高），减轻因数值误差导致的全图误匹配（如 `finalize_results` 中与整图框的 IoU 比较）。模块仅含纯张量运算，无可学习参数。

## 逐段说明

### `adjust_bboxes_to_image_border(boxes, image_shape, threshold=20)`（行 4）

**参数**：`boxes` 为 `(N,4)` xyxy 格式（原地修改）；`image_shape` 为 `(h,w)`；`threshold` 默认 20 像素。

**行 16–17**：解包高度 `h`、宽度 `w`。

**行 19–22**：按轴判断 proximity：
- `boxes[:,0] < threshold` 时将 `x1` 置 0；
- `y1` 同理置 0；
- `x2 > w - threshold` 时置 `w`；
- `y2 > h - threshold` 时置 `h`。

**行 23**：返回修改后的 `boxes` 引用。