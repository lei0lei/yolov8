# `ultralytics/models/yolo/world/train.py`

`WorldTrainer` 继承 `DetectionTrainer`：用 `WorldModel`、训练 batch 注入缓存文本嵌入；`nc` 语义为单图最大不同文本槽位（≤80）。含模块级回调 `on_pretrain_routine_end`。

## 逐段说明

### `on_pretrain_routine_end`（20–24）

从 `test_loader.dataset.data["names"]` 取类名（`/` 前段），`trainer.ema.ema.set_classes(names)`，各 rank 均需执行。

### `__init__`（56–68）

禁止 `compile=True`；`text_embeddings=None`。

### `get_model`（70–97）

`WorldModel`：`nc=min(data["nc"],80)`；load weights；注册 `on_pretrain_routine_end` 回调（去重 append）。

### `build_dataset`（99–116）

`multi_modal=(mode=="train")`；train 时 `set_text_embeddings([dataset], batch)` 缓存类名嵌入。

### `get_class_counts`（118）

绑定 `BaseTrainer.get_class_counts`（cls 通道为 prompt 位非数据集类）。

### `set_text_embeddings`（120–143）

遍历数据集 `category_names`，`generate_text_embeddings` 合并到 `self.text_embeddings`。

### `generate_text_embeddings`（145–171）

文本编码器：YOLOE 用 `text_model` 属性否则 `clip:ViT-B/32`；缓存路径 `text_embeddings_{model}.pt`；键序与 texts 一致则直接加载；否则 `get_text_pe`（YOLOE 传 `without_reprta=True`），`zip` 存 dict 并 `torch.save`。

### `preprocess_batch`（173–182）

先 `DetectionTrainer.preprocess_batch`；展平 `batch["texts"]`；stack 缓存嵌入；reshape 为 `(B, n_text, dim)` 写入 `txt_feats`。