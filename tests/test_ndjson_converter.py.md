# `tests/test_ndjson_converter.py`

本文件单元测试 `convert_ndjson_to_yolo` 对深度与检测任务的 NDJSON 清单转换行为。通过本地 `ThreadingHTTPServer` 提供带 query 签名的图片与深度 PNG，模拟平台签名 URL 下载。覆盖默认/自定义深度缩放、完成标记缓存复用、残缺配对清理、缺失 depth URL 校验，以及检测任务下 fraction/split 子集选择与 UTF-8 数据集名保留。

## 逐段说明

### `_QuietHandler`（第 18–22 行）

继承 `SimpleHTTPRequestHandler`，覆盖 `log_message` 为空，避免 HTTP 访问刷屏 pytest 输出。

### `_write_manifest`（第 25–51 行）

写入两行图像记录（train/val split）及可选 `depth_scale`、缺失深度 URL 开关；生成 `depth.ndjson` 风格 manifest 供各测试复用。

### `depth_server` fixture（第 54–72 行）

在 `tmp_path/source` 写入 train/test 的 jpg 与对齐形状的深度 PNG；绑定随机端口的守护线程 HTTP 服务；yield `(base_url, depth_array)`；teardown shutdown 并 join。

### `test_convert_depth_ndjson_downloads_image_target_pairs`（第 75–93 行）

转换后断言 YAML：`task=depth`、`nc=1`、`depth_scale=1000`、路径布局 `images/{split}` 与 `depth/{split}`，无 `labels/`；两 split 各一张图，深度与源数组_allclose。

### `test_convert_depth_ndjson_preserves_scale`（第 96–104 行）

manifest 带 `depth_scale=256` 时，生成 data yaml 中 `depth_scale` 为 256。

### `test_convert_depth_ndjson_reuses_existing_conversion`（第 107–124 行）

首次 `fraction=[1,1]` 转换；第二次 monkeypatch `YAML.save` 为 fail，仍应返回同一路径（缓存命中）。删除 val 深度图并去掉 yaml 中 `complete` 后再次转换应修复文件且路径不变。

### `test_convert_depth_ndjson_removes_incomplete_pair`（第 127–139 行）

val 深度 URL 指向不存在资源；期望 `RuntimeError` 匹配 `Downloaded 1/2 images`；数据集目录不应残留 val 图像/深度或 `data.yaml`。

### `test_convert_depth_ndjson_rejects_missing_url`（第 142–158 行）

`depth` 记录无 `url` 字段；`convert_ndjson_to_yolo` 应抛 `ValueError` 匹配 `missing depth.url`，不发 HTTP。

### `test_convert_ndjson_selects_split_fractions`（第 161–197 行）

构造 10 张 detect 清单（一张 split=test）。多组 `(fraction, split, expected_test)` 断言 train/val/test 文件名集合；`name` 与中文 `数据集` 保留；`fraction=0.05` 时 train 至少含 `1.jpg`（避免空 split）。