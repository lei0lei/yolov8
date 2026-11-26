# `ultralytics/trackers/deep_oc_sort.py`

Deep OC-SORT 组合 OC-SORT 观测关联、BoT-SORT 式外观 min 融合与 GMC。`DeepOCSortTrack` 在 OCSortTrack 上增加置信度自适应 EMA 嵌入；`DeepOCSortTrack.multi_gmc` 仅旋转 XYAH 的 x,y 与 vx,vy，保持 a,h 不变并 warp `last_observation`；`DeepOCSORT` 默认 `use_byte=False`。

## 逐段说明

### 类 `DeepOCSortTrack`

- `__init__`（31～59）：`alpha_fixed_emb`、`det_thresh`，可选初始特征。
- `update_features`（61～80）：高于 `det_thresh` 时 `trust` 调制 alpha，否则 alpha=1 忽略低分嵌入。
- `update`/`re_activate`（82～103）：有新 `curr_feat` 则平滑后父类更新。
- `multi_gmc`（105～157）：自定义 8×8 旋转块；变换观测 xyxy 中心与宽高。

### 类 `DeepOCSORT`

- `__init__`（170～189）：`GMC`、`proximity_thresh`、`appearance_thresh`、`build_encoder`。
- `init_track`（191～229）：encoder 存在时对每检测抽特征构造 `DeepOCSortTrack`。
- `_pre_first_associate`（231～246）：`gmc.apply` 后 `DeepOCSortTrack.multi_gmc`。
- `_fuse_appearance`（248～262）：`embedding_distance/2`，超 appearance 或 IoU 掩膜处置 1.0，与运动代价 `minimum`。
- `reset`（264～267）：父类 reset + `gmc.reset_params`。