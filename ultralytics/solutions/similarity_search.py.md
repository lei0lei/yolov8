# `ultralytics/solutions/similarity_search.py`

`VisualAISearch` 用 CLIP（ViT-B/32）对目录图像建 L2 归一化嵌入索引，文本查询做内积检索。`SearchApp` 封装 Flask：`templates/similarity-search.html` 表单 POST 查询并展示静态图 URL。

## 逐段说明

### `VisualAISearch.__init__`（第 45～71 行）

要求 torch≥2.4；缺目录下载 assets；`build_text_model`+`load_or_build_index`。

### `extract_image_feature` / `extract_text_feature`（第 73～79 行）

`encode_image`/`encode_text` 转 numpy。

### `load_or_build_index`（第 93～124 行）

遍历图像扩展名，跳过失败；`vstack` 后 `_normalize` 得 `index`。

### `search`（第 126～152 行）

归一化文本特征，矩阵乘得分，top-k 过滤 `similarity_thresh`，日志排名。

### `__call__`（第 154～156 行）

委托默认参数 `search`。

### 类 `SearchApp`（第 159～212 行）

- `__init__`：Flask，`static_folder` 指向图像目录，`/`→`index`。
- `index`：POST 时 `searcher(query)` 渲染模板。
- `run`：`app.run(debug=...)`。