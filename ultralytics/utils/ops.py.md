# `ultralytics/utils/ops.py`

本模块集中实现训练、验证与推理中的几何与张量后处理：检测框格式互转、LetterBox 反变换、实例/语义掩码解码与缩放、切片推理拼图合并，以及匈牙利匹配的线性分配求解。`Profile` 提供与 GPU/NPU/XPU 同步的计时代码。多数函数同时支持 NumPy 与 PyTorch，并在 MPS/macOS 14 上对 `clamp_` 做兼容分支。`merge_tiles` 在滑窗或瓦片预测后合并重叠区域的框、掩码与 dense 输出，是大幅面推理路径的核心。

## 逐段说明

### 类 `Profile`（第 20～76 行）

继承 `ContextDecorator`，可作为装饰器或 `with` 上下文统计耗时。`__init__` 保存初始累计时间 `t`、可选 `device`；若设备类型为 cuda/npu/xpu，则 `accelerator` 为对应后端模块，否则为 `None`。`__enter__` 记录 `start = self.time()`；`__exit__` 计算 `dt` 并累加到 `t`。`time()` 在存在加速器时先 `synchronize` 再读 `perf_counter`。`__str__` 返回累计秒数描述。

### `merge_tiles`（第 78～114 行）——整图捷径与 dense 合并

入参为各瓦片后处理后的 `dict`、窗口 `windows`、整图 `shape`、NMS 的 `iou`/`agnostic`/`max_det`。单窗且覆盖全图时直接返回 `first`。若首元素含 `probs`（分类），对多瓦片 `torch.stack(...).mean(0)`。若无 `bboxes`，按键取语义图或深度：`semantic_mask` 用距离边缘的权重与已有权重取 max 融合类别；连续图加权平均再除以权重和。

### `merge_tiles`（第 115～163 行）——检测框拼接与 NMS

拼接各瓦片 `bboxes`、`conf`、`cls` 等；按窗口左上角偏移平移 xyxy 或旋转框角点；根据是否贴内边缘识别切碎片段，若另一窗口能完整看见则丢弃碎片。对有效索引做类偏移后的 `nms_boxes`；对未被保留碎片类再补回并二次 NMS。关键点按偏移还原；掩码按采样步长 `r` 将 box-local ROI 拼到全图 `clip_boxes` 返回。

### `segment2box`（第 166～211 行）

空 segment 返回四维零。完全在图内：直接 `min/max` 得 xyxy。完全在图外：零框。跨边界：用边与段求交点，并入多边形内可见顶点与在轮廓内的角点，再取 min/max；无点则零框。

### `scale_boxes`（第 214～258 行）

`ratio_pad` 缺省时用 LetterBox 的 `gain` 与 `pad` 从 `img1_shape` 推算；否则用给定元组。`padding=True` 时先减 pad；四通道按 `gain_x/gain_y` 缩放；`xywh` 为真则返回未 clip 的框，否则 `clip_boxes` 到 `img0_shape`。

### `make_divisible`（第 261～273 行）

`divisor` 为张量时取 `max` 转 int；返回 `ceil(x/divisor)*divisor`。

### `clip_boxes` / `clip_coords`（第 276～326 行）

从 `shape` 取 `h,w`。Torch：非 macOS14 且非 MPS<2.5 用就地 `clamp_`，否则赋值式 `clamp`；NumPy 对 x/y 轴分别 `clip`。

### `xyxy2xywh` / `xywh2xyxy`（第 329～372 行）

list/tuple 先转 `float32` ndarray；断言末维为 4；`empty_like` 分配输出；中心与宽高按公式写入。

### `xywhn2xyxy` / `xyxy2xywhn`（第 375～423 行）

归一化与像素互转；`xyxy2xywhn` 可选先 `clip_boxes`。

### `xywh2ltwh` / `xyxy2ltwh` / `ltwh2xywh` / `ltwh2xyxy`（第 426～551 行）

各格式间复制张量或数组后改对应下标，实现左上宽高与中心点格式转换。

### `xyxyxyxy2xywhr`（第 471～504 行）

逐多边形 `cv2.minAreaRect` 得中心、边与角度；规范化为长边为 `w`、角度在 [-π/4,3π/4)；Torch 输入则结果转回同 device/dtype 张量。

### `xywhr2xyxyxyxy`（第 507～536 行）

用 cos/sin 旋转向量构造四角；兼容 NumPy 与 Torch 的 `cat/stack`。

### `segments2boxes`（第 554～567 行）

每段 min/max 得 xyxy 再 `xyxy2xywh` 批量转换。

### `resample_segments`（第 570～590 行）

闭合段插值到 `n` 点；长度已等于 `n` 跳过；否则闭合、线性插值 x/y。

### `crop_mask`（第 593～613 行）

框与掩码设备对齐；用行列索引广播乘法就地掩掉框外像素，避免物化全尺寸 bool 网格。

### `letterbox_span`（第 616～628 行）

按与 `scale_boxes` 一致的 gain 计算内容区在 letterbox 画布中的 `top,left,ch,cw`。

### `process_mask`（第 631～659 行）

无检测返回空 uint8 栈。系数与 prototype 相乘得 logits；`upsample` 时先双线性到输入尺寸再 crop，否则缩框到 prototype 分辨率；`gt_(0).byte()` 后 `crop_mask`。

### `process_mask_native`（第 662～689 行）

分批上采样与阈值，控制峰值显存；`scale_masks` 对齐 letterbox；最终 `crop_mask`。

### `process_mask_local`（第 692～751 行）

按框像素在 prototype 网格双线性采样，批量合并相似面积框以控内存；输出 `(rois, masks, shape)` 三元组与 `keep` 掩码。

### `scale_masks`（第 754～797 行）

尺寸已匹配直接返回；空通道维返回零张量；从 `letterbox_span` 或 `ratio_pad` 裁内容区再 `interpolate`；无 padding 且尺度误差时用 `affine_grid`+`grid_sample` 精确采样。

### `scale_coords`（第 800～835 行）

与 `scale_boxes` 类似的 pad/gain 处理前两个坐标通道；`clip_coords`；可选除以 `img0` 宽高归一化。

### `regularize_rboxes`（第 838～853 行）

`t % π` 判断并交换宽高，角度规范到 [0, π/2)。

### `masks2segments`（第 856～886 行）

逐掩码 `findContours`；`all` 策略合并多轮廓，`largest` 取最长；无轮廓返回 (0,2) float32。

### `convert_torch2numpy_batch`（第 889～898 行）

BCHW float → BHWC uint8，×255 clamp 后 CPU numpy。

### `clean_str` / `empty_like`（第 901～915 行）

正则替换非法文件名字符；`empty_like` 保持 dtype 创建空容器。

### `linear_sum_assignment`（第 921～965 行）

首次调用懒加载 SciPy 求解器，失败则用 `_linear_sum_assignment_numpy`；输入转 float64 调用后端。

### `_linear_sum_assignment_numpy`（第 968～1013 行）

矩形矩阵 Jonker-Volgenant：必要时转置使行≤列；对 `+inf` 不可行抛 `ValueError`；对行索引稳定排序以匹配 SciPy 输出约定。