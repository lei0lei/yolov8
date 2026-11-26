# `ultralytics/nn/modules/head.py`

本文件实现 YOLO 家族各类任务的输出头：检测、实例分割、旋转框、姿态、深度、分类、语义分割，以及开放词汇（World/YOLOE）、RT-DETR 与 YOLOv10 专用头。公共基类 `Detect` 负责多尺度特征上的框 DFL 回归与分类，并支持 **one2many（训练分配）** 与 **one2one（推理/NMS-free）** 双分支。训练前向返回原始 logits 字典供 TAL/损失使用；推理前向经锚点解码、sigmoid、可选 `postprocess` top-k，导出时返回单张量。分割头附加 **mask 系数** 与 **Proto**；姿态头附加 **关键点**；OBB 附加 **角度**；深度头输出 **log-depth/米制深度图**。

## 通用前向约定（Detect 系）

| 模式 | 返回 |
|------|------|
| `training=True` | `dict`：`boxes` `(B,4*reg_max,N)`、`scores` `(B,nc,N)`，可选 `mask_coefficient`/`angle`/`kpts`/`proto`；双头时为 `one2many`/`one2one` 嵌套 |
| `training=False`, 非 export | `(y, preds)`：`y` 推理张量；`preds` 原始 dict |
| `export=True` | 仅 `y`；分割等多模态为 `(y, proto)` 等 |
| end2end 推理 | `y` 经 `postprocess` → `(B,max_det,6+extra)`：`[x1,y1,x2,y2,score,cls,...]` |
| 标准推理 `y` | `(B,4+nc+extra,N)`：解码框 + sigmoid 类别 + 任务 extra 通道 |

---

## 逐段说明

### 模块导入与 `__all__`（第 1–35 行）

从 `block`/`conv`/`transformer` 引入 DFL、Proto、对比头等；`__all__` 列出对外检测头类名。

### `Detect` 类（第 38–296 行）

YOLO 核心检测头：`cv2` 框分支、`cv3` 类分支、`dfl` 分布积分；`end2end` 时深拷贝为 `one2one_cv2/cv3`。

#### 类属性（第 80–89 行）

`dynamic/export/format/max_det/agnostic_nms/shape/anchors/strides/legacy/xyxy` 等运行期与导出标志。

#### `_grouped_topk`（第 91–102 行，静态）

长轴分组 top-k，缓解 TensorRT 等设备上大 dim topk。

#### `_gather`（第 104–106 行）

沿 anchor 维按 index 取行。

#### `__init__`（第 108–143 行）

`no = nc + reg_max*4`；每层 `cv2/cv3` 结构（`legacy` 与否决定 `cv3` 是否 DWConv）；`end2end` 复制 one2one 头。

#### `one2many` / `one2one` 属性（第 145–153 行）

返回 `box_head`/`cls_head` 字典供 `forward_head` 解包。

#### `end2end` 属性与 setter（第 155–167 行）

存在 `one2one_cv2` 且 `_end2end` 或 `cv2 is None`（融合后）时为真；setter 只改推理选用分支。

#### `forward_head`（第 169–178 行）

多尺度 concat：`boxes` `(B,4*reg_max,ΣHW)`、`scores` `(B,nc,ΣHW)`、`feats` 原特征列表。

#### `forward`（第 180–204 行）

**训练**：`forward_head(one2many)`；若有 one2one 再算 `one2one`（训练时特征 detach），返回嵌套 dict。

**推理**：选 `one2one` 或 `one2many` → `_inference` → 若 `end2end` 则 `postprocess(permute)`；export 仅 `y`，否则 `(y, preds)`。

#### `_inference`（第 206–217 行）

`dbox = _get_decode_boxes`；`cat(dbox, scores.sigmoid(), dim=1)` → `(B,4+nc,N)`。

#### `_get_decode_boxes`（第 219–227 行）

必要时 `make_anchors`；`dfl(boxes)` + `decode_bboxes` * strides。

#### `bias_init`（第 229–241 行）

框末层 bias=2.0；类 bias 按 stride 与 `nc` 对数先验。

#### `decode_bboxes`（第 243–247 行）

`dist2bbox`；OBB 子类可传 `angle`。

#### `postprocess`（第 249–264 行）

拆 boxes/scores/extra；`get_topk_index`；gather 框与 extra → `(B,k,6+extra)`。

#### `get_topk_index`（第 266–288 行）

类别相关或 agnostic top-k；export engine 可用 `groups=8`。

#### `fuse`（第 290–295 行）

按 `end2end` 删除未用分支（`one2one_*` 或 `cv2/cv3` 置 None）。

---

### `Segment` 类（第 298–390 行）

在 Detect 上增加 `Proto` 与每层 `cv4` mask 系数。

#### `__init__`（第 319–346 行）

`proto = Proto(ch[0], npr, nm)`；`cv4` → `nm` 通道；end2end 复制 `one2one_cv4`。

#### `forward`（第 358–375 行）

**训练**：`super().forward` 得 dict，写入 `proto`（one2one 用 detach proto），返回 preds。

**推理**：`(y, proto)` 或 `((y,proto), preds)` / export `(y,proto)`。`y` 在 `_inference` 含 **nm 维 mask 系数**。

#### `_inference`（第 377–380 行）

`cat(preds, mask_coefficient, dim=1)` → `(B,4+nc+nm,N)`。

#### `forward_head`（第 382–390 行）

增加 `mask_coefficient` `(B,nm,N)`。

---

### `Segment26` 类（第 393–461 行）

#### `__init__`（第 414–434 行）

`Proto26(ch, ...)` 替代 `Proto`；多尺度 proto。

#### `forward`（第 436–455 行）

同 Segment，proto 可为 tuple；one2one proto 逐项 detach。

#### `fuse`（第 457–461 行）

`super().fuse()` + `proto.fuse()`。

---

### `OBB` 类（第 464–535 行）

#### `__init__`（第 484–502 行）

`cv4` 预测 `ne` 角参数；one2one 复制。

#### `_inference`（第 514–517 行）

拼接 **angle** → `(B,4+nc+ne,N)`。

#### `forward_head`（第 519–531 行）

`angle` sigmoid 映射到 `[-π/4,3π/4]` 写入 dict。

#### `decode_bboxes`（第 533–535 行）

`dist2rbox(bboxes, angle, anchors)`。

---

### `OBB26` 类（第 538–569 行）

#### `forward_head`（第 558–569 行）

角输出 **不经 sigmoid**，原始 logits 供 YOLO26 损失/解码。

---

### `Pose` 类（第 572–660 行）

#### `__init__`（第 593–617 行）

`kpt_shape`、`nk`；`cv4` 输出 `nk` 维。

#### `_inference`（第 629–632 行）

`kpts_decode` 后拼接 → `(B,4+nc+nk,N)`。

#### `forward_head`（第 634–642 行）

`kpts` 原始 `(B,nk,N)`。

#### `kpts_decode`（第 644–660 行）

相对锚点 *2 + 锚点偏移，乘 stride；3 维时可见度 sigmoid。export  reshape 为 `(B,nk,N)`。

---

### `Pose26` 类（第 663–782 行）

#### `__init__`（第 684–714 行）

`RealNVP` flow（训练）；`cv4` 仅特征；`cv4_kpts`/`cv4_sigma` 分支；one2one 全套复制。

#### `one2many` / `one2one`（第 716–736 行）

含 `pose_head`、`kpts_head`、`kpts_sigma_head`。

#### `forward_head`（第 738–757 行）

训练额外输出 `kpts_sigma`；推理仅用 `kpts`。

#### `fuse`（第 759–763 行）

删 flow 与 sigma 头。

#### `kpts_decode`（第 765–782 行）

YOLO26 锚点公式 `(y+anchors)*strides`，无 `*2` 偏移。

---

### `Depth` 类（第 785–882 行）

单通道密集深度，无检测框。

#### `__init__`（第 809–836 行）

金字塔 `proj`/`refine`/`head`；`cal_a`/`cal_b` log 仿射标定 buffer。

#### `log_depth`（第 838–858 行）

自顶向下融合 P3–P5，输出 **clamped log-depth** `(B,1,H/4,W/4)`。

#### `forward`（第 860–882 行）

**训练**：`{"depth": exp(out)}` 正深度供损失。

**推理**：`exp(cal_a*out+cal_b)`；**export** 双线性 4× 上采样至输入尺寸；litert 加 noop ReLU。

---

### `Classify` 类（第 885–944 行）

#### `__init__`（第 909–925 行）

Conv → GAP → Linear → `c2` 类。

#### `forward`（第 927–944 行）

**训练**：logits `(B,c2)`。

**推理**：softmax 概率；export 仅概率，否则 `(prob, logits)`。

---

### `WorldDetect` 类（第 947–1020 行）

文本嵌入对比分类，替代独立 `cv3` 类卷积。

#### `__init__`（第 969–991 行）

`cv3` 出 embed 维；`cv4` 为 `ContrastiveHead` 或 BN 版。

#### `forward`（第 993–1015 行）

每层 `cat(box, contrastive(cv3(f), text))` → 等效 `no` 通道；**训练** 返回单 dict；**推理** 同 Detect `_inference`。

#### `bias_init`（第 1017–1020 行）

仅初始化 `cv2` 框 bias。

---

### `LRPCHead` 类（第 1023–1101 行）

Prompt-free 提案过滤 + 词汇线性分类。

#### `__init__`（第 1047–1060 行）

enabled 时 `vocab` 1×1 conv 转 Linear。

#### `conv2linear`（第 1062–1069 行，静态）

权重复制到 `nn.Linear`。

#### `forward`（第 1071–1101 行）

`conf==0`：全 anchor；否则 `pf` sigmoid 阈值 + batch union mask；`vocab` 在保留锚点上打分，被滤 anchor 置 `-inf`。返回 `loc_feat`、`cls`、`mask`。

---

### `YOLOEDetect` 类（第 1104–1348 行）

开放词汇检测：BN 对比头、文本/视觉 prompt、`fuse` 将文本嵌入写入 conv 权重。

#### `__init__`（第 1136–1178 行）

`reprta`、`savpe`；end2end 复制 cv3/cv4。

#### `fuse`（第 1181–1202 行）

无 `txt_feats` 时同 Detect；否则 `_fuse_tp` 后删 `reprta` 为 Identity。

#### `_fuse_tp`（第 1204–1239 行）

逐层 fuse conv+bn，用文本矩阵重写 conv 权重与 bias。

#### `get_tpe` / `get_vpe`（第 1241–1252 行）

文本 MLP 归一化；视觉 SAVPE 或空张量。

#### `forward`（第 1254–1266 行）

有 `lrpc` 走 `forward_lrpc`，否则 `super().forward`（需 4 元输入含 text）。

#### `forward_lrpc`（第 1268–1295 行）

逐层 LRPC；拼 boxes/scores/index；`_inference` + 可选 end2end postprocess。

#### `forward_mask`（第 1297–1299 行）

检测默认可扩展为空 dict。

#### `_get_decode_boxes`（第 1301–1306 行）

有 `index` 时裁减 anchor 维。

#### `forward_head`（第 1318–1330 行）

`len(x)==4`：前三层图 + `x[-1]` text `(B,nc,embed)`；`contrastive_head` 得 scores。

#### `bias_init`（第 1332–1348 行）

未 fuse 时初始化对比头 bias；fuse 后走 Detect。

---

### `YOLOESegment` 类（第 1351–1471 行）

YOLOE + 实例分割。

#### `__init__`（第 1374–1405 行）

`Proto` + `cv5` mask 系数。

#### `forward_mask`（第 1422–1426 行）

LRPC 时按 `index` 裁 mask 系数。

#### `forward` / `_inference` / `forward_head`（第 1428–1465 行）

同 Segment 与 YOLOE 组合逻辑。

#### `fuse`（第 1467–1471 行）

`super().fuse` + 可选 `proto.fuse()`。

---

### `YOLOESegment26` 类（第 1474–1538 行）

#### `__init__`（第 1498–1518 行）

直接 `YOLOEDetect.__init__` + `Proto26`。

#### `forward`（第 1520–1538 行）

`Proto26(x, return_semantic=False)`；其余同 YOLOESegment。

---

### `RTDETRDecoder` 类（第 1541–1890 行）

Transformer 检测头，非 anchor-based。

#### `__init__`（第 1589–1666 行）

`input_proj`、可变形 `decoder`、CDN 去噪嵌入、`enc_*`/`dec_*` 头、`query_pos_head`；`_reset_parameters`。

#### `forward`（第 1668–1718 行）

**训练**：`_get_encoder_input` → `get_cdn_group` → `_get_decoder_input` → `decoder` → 返回 `(dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)`。

**推理**：`postprocess` on 最后一层 `dec_bboxes/scores` → `(B,k,6)` **cx,cy,w,h,score,cls**；export 仅 `y`。

#### `postprocess`（第 1720–1742 行）

flatten scores top-k；gather 对应 query 框。

#### `_generate_anchors`（第 1744–1777 行，静态）

多尺度 grid 锚，logit 空间 + valid_mask。

#### `_get_encoder_input`（第 1779–1803 行）

投影并 flatten concat → `(B,ΣHW,hd)`。

#### `_get_decoder_input`（第 1805–1870 行）

编码器 token 打分 top-k 选 query；`enc_bbox_head` 残差 + 锚；可选 CDN concat；返回 embed、refer_bbox、enc_bboxes、enc_scores。

#### `_reset_parameters`（第 1872–1890 行）

分类/回归头与投影 Xavier/constant 初始化。

---

### `v10Detect` 类（第 1893–1936 行）

#### `__init__`（第 1918–1936 行）

`super(..., end2end=True)`；轻量 `cv3`/`one2one_cv3` 结构。

继承 Detect 双头训练与 one2one 推理 postprocess。

---

### `SemanticSegment` 类（第 1939–2005 行）

每像素语义类，无框。

#### `__init__`（第 1960–1976 行）

`classifier` on P3；可选 P4 `aux_head`。

#### `forward`（第 1978–2005 行）

**训练**：主 logits `(B,nc,H/8,W/8)`，可有 aux 元组。

**推理**：返回 logits。

**export**：8× 上采样；部分格式 **bake argmax** → `(B,H,W)` uint8/int32 类图。