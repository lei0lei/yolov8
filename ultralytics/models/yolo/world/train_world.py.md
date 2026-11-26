# `ultralytics/models/yolo/world/train_world.py`

`WorldTrainerFromScratch`：混合 YOLO 检测集与 grounding JSON 从零训 YOLO-World；重写 `get_dataset`/`build_dataset`，跳过标签分布图，final_eval 指向单一 val yaml。

## 逐段说明

### `__init__`（55–68）

`super(WorldTrainer)`。

### `build_dataset`（70–102）

val：`build_yolo_dataset` rect=False。
train：列表项为 str 则 `build_yolo_dataset(..., multi_modal=True)`；dict 则 `build_grounding(img_path, json_file, max_samples=data["nc"])`。
`set_text_embeddings`；多集 `YOLOConcatDataset`。

### `check_data_config`（104–119）

非 dict 则 `YAML.load(check_file)`。

### `get_dataset`（121–181）

train/val 须为非空 dict（含 yolo_data/grounding_data）。
`check_det_dataset` 各 yaml；val 仅允许一个；LVIS 用 minival split。
合并 train/val 路径列表 + grounding 路径解析（相对 `DATASETS_DIR`）。
`nc/names/path/channels` 取自唯一 val 配置；`training_data` 按各 train yaml 的 train 路径索引子 dict；single_cls 覆写。

### `plot_training_labels`（183–184）

空实现跳过。

### `final_eval`（186–194）

`validator.args.data` 设为 `args.data["val"]["yolo_data"][0]`；split minival（LVIS）或 val；`super().final_eval()`。