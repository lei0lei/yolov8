# `ultralytics/data/split.py`

本文件实现两类数据集划分工具：分类数据按类文件夹复制为 train/val 树（`split_classify_dataset`），以及检测类数据按权重随机写入 `autosplit_*.txt` 列表（`autosplit`）。对外接口即上述两个函数；`split_classify_dataset` 在 `utils.check_cls_dataset` 发现缺少 `train/` 子目录但存在散落图像时被自动调用。`autosplit` 多用于 COCO 式「仅 images 目录 + 文本列表」工作流，默认路径指向 `DATASETS_DIR/coco8/images`。常量与路径派生复用 `utils.IMG_FORMATS`、`img2label_paths`。

## 逐段说明

### 导入区（L1–L10）

- **标准库**：`random`（固定种子 0 的可复现划分）、`shutil.copy2`（保留元数据复制）、`Path`。
- **`ultralytics.data.utils`**：`IMG_FORMATS` 过滤合法图像扩展名；`img2label_paths` 由图像路径推导 YOLO `.txt` 标签路径。
- **`ultralytics.utils`**：`DATASETS_DIR` 作 `autosplit` 默认根；`LOGGER`、`TQDM` 打日志与进度。

### `split_classify_dataset`（L13–L100）

**职责**：在 `{source_dir}_split` 下新建 `train/`、`val/`，按类子目录、固定比例复制图像，不删除源数据。

#### 目录准备（L64–L72）

- **L64–L66**：`split_path = Path(f"{source_path}_split")`；train/val 子路径确定。
- **L70–L72**：仅 `mkdir(exist_ok=True)`，**不** `rmtree`：注释说明 `check_cls_dataset` 会多次、多进程触发，删除会破坏 `*.cache` / `*.npy` 并产生竞态。

#### 按类划分（L74–L97）

- **L75**：`source_path` 下所有**一级**子目录视为类。
- **L76–L78**：日志输出类数与 train/val 比例。
- **L81–L84**：每个类在 train/val 下创建同名子目录。
- **L87**：`rglob` 收集该类下扩展名属于 `IMG_FORMATS` 的文件并排序。
- **L89**：`random.Random(0).shuffle` — 与全局 `random` 隔离，保证重复运行划分一致。
- **L90**：`split_idx = int(len * train_ratio)`，前 `split_idx` 张进 train。

**复制循环（L92–L97）**

- **L93**：索引 `< split_idx` → `target=train`，否则 `val`；`previous` 为另一侧，用于清理旧位置副本。
- **L94**：相对路径保留类名与子路径结构（支持类内子文件夹）。
- **L95**：若图像因比例变化需从 val 挪到 train，删除另一侧旧文件，避免重复。
- **L96–L97**：确保目标父目录存在，`copy2` 复制。

- **L99–L100**：汇总日志并返回 `split_path`（供 `check_cls_dataset` 把 `data_dir` 指向新目录）。

**谁调用**：`ultralytics/data/utils.py` 的 `check_cls_dataset`，当无 `train/` 但有图像时。

### `autosplit`（L103–L137）

**职责**：对 `path` 下全部图像按 `weights` 抽样分到 train/val/test，路径写入 `path.parent` 下三个 txt。

#### 收集与分配（L123–L126）

- **L123**：`path` 转为绝对路径，保证 `img2label_paths` 与训练加载器路径一致。
- **L124**：递归收集 `IMG_FORMATS` 图像并排序。
- **L126**：`random.Random(0).choices([0,1,2], weights=weights, k=n)` — 每张图独立按权重抽 split 索引；可复现。

#### 写列表文件（L128–L137）

- **L128–L131**：三个文件名固定；若父目录已存在同名文件先删除，避免追加污染。
- **L133**：日志；`annotated_only` 时在消息中说明仅使用有标签的图。
- **L134–L137**：`zip(indices, files)` 遍历：
  - **L135**：若 `annotated_only`，要求 `img2label_paths([img])[0]` 存在，否则跳过。
  - **L136–L137**：追加一行 `./{相对 path.parent 的 posix 路径}`，供 YOLO 数据集 yaml 引用。

**格式**：输出为纯文本路径列表，每行一张图，前缀 `./`，无标签内容。

### `__main__`（L140–L141）

- 直接运行模块时以硬编码 `"caltech101"` 调用 `split_classify_dataset`，便于本地快速试验。