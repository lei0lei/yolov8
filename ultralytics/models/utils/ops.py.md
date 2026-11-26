# `ultralytics/models/utils/ops.py`

实现 DETR/RT-DETR 训练所需的匈牙利双分图匹配与对比去噪（CDN）查询构造。`HungarianMatcher` 按分类、L1 框与 GIoU 加权成本矩阵在 CPU 上 `linear_sum_assignment`。`get_cdn_group` 在非训练或禁用条件返回四元 None；否则对 GT 加噪生成去噪 query 与自注意力掩码及 `dn_meta`。

## 逐段说明

### `HungarianMatcher.__init__(...)`（行 44）

- **行 61–62**：`cost_gain` 默认 `class:1,bbox:5,giou:2`。
- 保存 `use_fl`、`alpha`、`gamma`。

### `HungarianMatcher.forward(...)`（行 68）

- **行 96**：读 `bs,nq`。
- **行 98–99**：`sum(gt_groups)==0` 时每张图返回空 long 张量配对。
- **行 102–103**：`pad_sequence` 按图 pad GT 框与类。
- **行 104–106**：`pred_scores` detach float；`use_fl` 则 sigmoid 否则 softmax；`pred_bboxes` detach float。
- **行 109**：`gather` 取各 GT 类对应 query 类维度得分。
- **Focal 分类代价（行 110–113）**：正负 focal 项相减得 `cost_class`；否则 **行 115** `-pred_scores`。
- **行 118**：L1 绝对差求和得 `cost_bbox`。
- **行 121**：`1 - GIoU` 得 `cost_giou`。
- **行 124–128**：按 `cost_gain` 加权求和为 `C`。
- **行 131**：NaN/Inf 置 0。
- **行 133–134**：`C` 转 CPU，每 batch 切片 `[:,:n]` 做匈牙利匹配。
- **行 135**：`gt_groups` 前缀和得全局 GT 偏移。
- **行 136–139**：组装 `(query_idx, gt_idx+offset)` 列表。

### `get_cdn_group(...)`（行 142）

**早退（行 187–188）**：非 training、`num_dn<=0` 或 `batch is None` → 四 None。

**行 189–192**：`max_nums = min(max(gt_groups), num_dn)`；为 0 则四 None。

**行 194**：`num_group = num_dn // max_nums`。

**行 196–199**：取 `gt_cls`、`gt_bbox`、`batch_idx`。

**大行数 GT 子采样（行 204–211）**：若某图 GT 数超 `max_nums`，每图 `randperm` 取前 `max_nums` 排序索引，重排 cls/bbox/b_idx 并收紧 `gt_groups`。

**行 214–217**：正负面各 `num_group` 倍复制 cls/bbox/b_idx。

**行 220**：`neg_idx` 指向每组中负样本块索引。

**类别噪声（行 222–228）**：`cls_noise_ratio>0` 时以 `cls_noise_ratio*0.5` 概率随机换类。

**框噪声（行 230–242）**：`xywh2xyxy`；`diff` 为半宽高×`box_noise_scale`；随机符号与 `rand_part`，负样本索引处 `rand_part+=1`；加噪 clip 0–1；`xyxy2xywh` 后 `logit` 逆 sigmoid。

**行 244–254**：更新总 `num_dn`；`class_embed[dn_cls]` 散射进 `padding_cls`/`padding_bbox`（按 `map_indices` 与 `dn_b_idx`）。

**注意力掩码（行 256–268）**：尺寸 `num_dn+num_queries`；检测 query 不能看去噪段；各去噪组间互不可见（分三种 `i` 情况填 True）。

**行 269–274**：`dn_meta` 含 pos/gt 索引列表、`dn_num_group`、`dn_num_split`。

**行 276–281**：将 padding 与 mask 转到 `class_embed.device` 返回四元组。