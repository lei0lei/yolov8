# `ultralytics/models/sam/amg.py`

本模块实现 Automatic Mask Generation（AMG）及多尺度裁剪推理所需的纯数值工具，无神经网络定义。函数涵盖：裁剪坐标与全图坐标的框/点/掩码互转、分层 crop 网格生成、均匀点网格采样、掩码稳定性打分、小连通域/孔洞清理，以及由二值掩码反推 XYXY 外接框。实现混合使用 PyTorch（张量批处理）与 NumPy/OpenCV（形态后处理），供 `predict` 或 solutions 中「全自动撒点分割」流程调用。命名与行为对齐 Meta 原始 SAM `automatic_mask_generator` 辅助逻辑，便于与大图分块、边缘框过滤策略组合。

## 逐段说明

### 导入（第 1–11 行）

- `annotations` 前向引用；`math` 用于 crop 长度；`Generator` 标注批迭代器；`product` 生成 crop 网格笛卡尔积；`numpy`/`torch` 为数值后端。

### 函数 `is_box_near_crop_edge`（第 14–40 行）

- **第 34–35 行**：把 `crop_box`、`orig_box` 转为与 `boxes` 同设备的 float 张量。
- **第 36 行**：`uncrop_boxes_xyxy` 将 crop 局部坐标还原到原图坐标系。
- **第 37 行**：`near_crop_edge`：各边是否与 crop 矩形四边在 `atol` 内接近（`isclose`，`rtol=0`）。
- **第 38 行**：`near_image_edge`：是否与**原图**外框四边接近。
- **第 39 行**：`logical_and(near_crop_edge, ~near_image_edge)`，保留「贴 crop 边但不贴原图边」的边（避免误删触及整图边缘的框）。
- **第 40 行**：`any(dim=1)` 得每个框是否任意一边满足条件的一维 bool 张量。

### 函数 `batch_iterator`（第 43–68 行）

- **第 65 行**：断言 `args` 非空且所有可迭代对象长度与 `args[0]` 相同。
- **第 66 行**：`n_batches = len // batch_size + (1 if 有余数 else 0)`。
- **第 67–68 行**：对每个 batch 索引 `b`，对各参数切片 `[b*batch_size : (b+1)*batch_size]`，yield 列表（多路输入对齐的一批）。

### 函数 `calculate_stability_score`（第 71–97 行）

- **语义**：用 `mask_threshold ± threshold_offset` 二值化同一 logits 掩码，高阈值掩码嵌在低阈值掩码内，交并比即为稳定性。
- **第 95 行**：`intersections`：像素和 `> (mask_threshold + threshold_offset)`，先按 H 维 `sum(..., dtype=int16)` 再按 W 维 `sum(..., int32)` 省内存。
- **第 96 行**：`unions`：同理阈值为 `mask_threshold - threshold_offset`。
- **第 97 行**：逐元素 `intersections / unions` 返回每掩码标量分数。

### 函数 `build_point_grid`（第 100–106 行）

- **第 102 行**：`offset = 1/(2*n_per_side)`，使点落在像素中心意义下的 [0,1] 均匀网格。
- **第 103 行**：单边 `n_per_side` 个 linspace 点。
- **第 104–105 行**：`tile` 成 `n_per_side×n_per_side` 的 x/y 网格。
- **第 106 行**：`stack` 后 `reshape(-1, 2)` 返回 N×2 归一化坐标。

### 函数 `build_all_layer_point_grids`（第 109–111 行）

- 对 `i in range(n_layers + 1)`，调用 `build_point_grid(int(n_per_side / (scale_per_layer**i)))`，层数越深每边点数按 `scale_per_layer` 幂次递减，返回 `list[np.ndarray]`。

### 函数 `generate_crop_boxes`（第 114–162 行）

#### 初始化与原图层（第 134–140 行）

- 空列表 `crop_boxes`、`layer_idxs`；`im_h, im_w = im_size`；`short_side = min(im_h, im_w)`。
- 先追加整图框 `[0,0,im_w,im_h]`，`layer_idxs` 为 0。

#### 内嵌 `crop_len`（第 142–144 行）

- `math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops)`：给定原边长、每边 crop 数与 overlap 像素，求每个 crop 窗口边长。

#### 分层循环（第 146–160 行）

- 每层 `i_layer`：`n_crops_per_side = 2**(i_layer+1)`（2、4、8…）。
- **第 148 行**：`overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))`，层越细 overlap 按公式缩小。
- **第 150–151 行**：对宽高分别 `crop_len` 得 `crop_w`、`crop_h`。
- **第 153–154 行**：起始坐标列表，步长 `(crop_w - overlap)*i` / `(crop_h - overlap)*i`。
- **第 157–160 行**：`product(x0, y0)` 枚举子窗口，`box = [x0,y0, min(x0+crop_w,im_w), min(y0+crop_h,im_h)]` 裁剪到图像内，`layer_idxs` 记 `i_layer+1`。

#### 返回（第 162 行）

- `(crop_boxes, layer_idxs)` 等长列表。

### 函数 `uncrop_boxes_xyxy`（第 165–172 行）

- **第 167–168 行**：取 crop 左上角 `(x0,y0)`，构造 shape `(1,4)` 偏移 `[x0,y0,x0,y0]`。
- **第 170–171 行**：若 `boxes` 为 3 维（含通道维），对 offset `unsqueeze(1)` 以便广播。
- **第 172 行**：`boxes + offset` 得原图 XYXY。

### 函数 `uncrop_points`（第 175–182 行）

- 逻辑同框：偏移 `[x0,y0]`，3 维点张量时对 offset 加维，返回平移后的点坐标。

### 函数 `uncrop_masks`（第 185–193 行）

- **第 188–189 行**：若 crop 即全图（左上 0,0 且右下等于 `orig_w, orig_h`），直接返回 `masks`。
- **第 191–192 行**：计算需在四周 padding 的量，`pad = (left, right, top, bottom)` 对应 `F.pad` 四元组 `(x0, pad_x-x0, y0, pad_y-y0)`。
- **第 193 行**：`pad(..., value=0)` 将 crop 大小掩码嵌入 `orig_h×orig_w` 画布。

### 函数 `remove_small_regions`（第 196–230 行）

- **第 215 行**：懒加载 `cv2`。
- **第 217 行**：`mode` 必须是 `"holes"` 或 `"islands"`。
- **第 218 行**：`correct_holes = (mode == "holes")`。
- **第 219 行**：`working_mask = (correct_holes ^ mask).astype(uint8)`，孔洞模式对掩码取反后做连通域（把孔洞变成前景统计）。
- **第 220–221 行**：8 连通 `connectedComponentsWithStats`，`sizes` 去掉背景标签 0。
- **第 222 行**：面积 `< area_thresh` 的组件标签列表 `small_regions`（标签 id 为 `i+1`）。
- **第 223–224 行**：若无小组件，返回原 `mask` 与 `modified=False`。
- **第 225 行**：`fill_labels = [0, *small_regions]` 默认要保留/填充的标签集。
- **第 226–228 行**：**islands 模式**：若不以小组件为填充目标，则反转逻辑——保留大区域标签；若全部都很小则保留最大连通域标签 `argmax(sizes)+1`。
- **第 229–230 行**：`np.isin(regions, fill_labels)` 生成新二值掩码，`modified=True`。

### 函数 `batched_mask_to_box`（第 233–275 行）

#### 空输入（第 247–248 行）

- `torch.numel(masks)==0` 时返回与 batch 维一致、末维为 4 的全零张量（避免 `max` 空张量报错）。

#### 形状归一化（第 250–253 行）

- 记录原 `shape`，取末两维 `h,w`；若维数>2 则 `flatten(0,-3)` 合并 batch/通道为 leading 维，否则 `unsqueeze(0)` 成 `(1,H,W)`。

#### 上下边界（第 254–259 行）

- `in_height = masks.amax(-1)` 每行是否有前景；`in_height_coords = in_height * arange(h)`，行内最大坐标为底边 `bottom_edges`；对非前景行加 `h` 后用 `min` 得上边 `top_edges`。

#### 左右边界（第 261–266 行）

- 对宽度维对称操作得 `right_edges`、`left_edges`。

#### 空掩码修正与复原形状（第 268–275 行）

- **第 270 行**：`empty_filter`：右<左或下<上表示空掩码。
- **第 271–272 行**：堆叠为 XYXY，`out * (~empty_filter).unsqueeze(-1)` 空框归零。
- **第 275 行**：若原 shape 维数>2则 `reshape(*shape[:-2], 4)`，否则返回 `out[0]`（单向量掩码）。