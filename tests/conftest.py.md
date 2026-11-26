# `tests/conftest.py`

本文件是 Ultralytics pytest 的全局挂钩与共享 fixture 定义。它扩展命令行为（`--slow`、`--export-env`），在收集阶段按标记过滤慢测或按导出格式环境 ID 跳过无关用例，并在 GitHub Actions 上将失败信息写入 Actions 摘要。`isolated_model` 与 `isolated_model_path` 为 xdist 并行导出测试提供每用例独立的权重副本，避免同名 `.onnx` 等中间文件互相覆盖。会话开始时固定随机种子；会话结束仅在 controller 进程清理 weights 目录下的导出产物与临时下载，避免 worker 竞态删除共享资产。

## 逐段说明

### `solution_assets`（第 12–28 行）

会话级 fixture：返回按名称懒下载 Solutions 资产的闭包 `get_asset`。缓存目录为 `WEIGHTS_DIR/solution_assets`，缺失时通过 `safe_download` 从 `ASSETS_URL` 拉取 `tests.SOLUTION_ASSETS` 中对应文件。

### `pytest_addoption`（第 31–38 行）

注册 `--slow`（默认不跑 `@pytest.mark.slow`）与 `--export-env`（仅运行属于指定隔离导出环境的 `test_exports.py` 项）。

### `_export_format_from_item`（第 41–54 行）

从收集到的 pytest item 推断其覆盖的导出格式：仅处理 `test_exports.py`；`test_torch2onnx_serializes_concurrent_exports` 映射为 `onnx`；`test_export_*` 名称最长匹配 `export_formats()` 中的格式键。

### `pytest_collection_modifyitems`（第 57–78 行）

无 `--slow` 时从 items 列表移除带 `slow` 关键字的项。指定 `--export-env` 时，对格式所属环境不匹配的导出测试添加 skip 标记。

### `pytest_runtest_logreport`（第 81–89 行）

在 GITHUB Actions 且非 xdist worker 时，对失败用例向 stderr 输出 `::error` 工作流命令，便于 PR 检查页展示 nodeid 与崩溃信息。

### `isolated_model_path`（第 92–103 行）

将模型复制到 `tmp_path` 下唯一路径；源权重不存在则 `attempt_download_asset` 后再复制。返回字符串路径供 `YOLO()` 使用。

### `pytest_sessionstart`（第 106–117 行）

调用 `init_seeds()` 统一测试会话随机性。

### `isolated_model`（第 120–131 行）

函数级 fixture：对共享 `tests.MODEL` 调用 `isolated_model_path`，供导出类测试避免文件名冲突。

### `pytest_sessionfinish`（第 134–153 行）

xdist worker 直接返回。controller 删除 cwd 与 `WEIGHTS_DIR` 下匹配的 onnx/torchscript 文件及 OpenVINO/CoreML 目录，防止多次 pytest 积累垃圾产物。