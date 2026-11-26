# `ultralytics/utils/tuner.py`

本模块实现 YOLO 的 Ray Tune 超参搜索集成：`run_ray_tune` 为入口，将模型放入 Ray 对象存储、定义可恢复的训练 trial、ASHA（或 BOHB 用 HyperBandForBOHB）调度与多种搜索算法适配。`_resolve_ray_search_alg` 及一组 `_create_*`/`_convert_*` 处理 Ax、BOHB、Nevergrad、ZOOpt、Optuna 等对 Tune `param_space` 形态的差异；`RAY_SEARCH_ALG_REQUIREMENTS` 映射各算法依赖包。

## 逐段说明

### `RAY_SEARCH_ALG_REQUIREMENTS`（第 8～18 行）

- 字典：算法名到 pip 包名或列表；`random` 为 None。

### `_get_ray_search_alg_kind`（第 21～46 行）

- `None` 返回 None；字符串 strip lower 空串则 None。
- 识别 searcher 类 `AxSearch`/`TuneBOHB`/`ZOOptSearch` 模块前缀返回对应 kind。

### `_validate_ax_search_space`（第 49～65 行）

- `check_requirements(ax-platform)`；`AxSearch.convert_search_space(space)` 返回参数列表。

### `_create_ax_search`（第 68～92 行）

- 用转换参数 `AxClient.create_experiment`，objective 为 `TASK2METRIC[task]` maximize；返回 `AxSearch(ax_client=...)`。

### `_convert_bohb_search_space`（第 95～150 行）

- flatten + `parse_spec_vars`；有 grid_vars 抛 ValueError。
- ConfigSpace 逐域添加 UniformFloat/Integer/Categorical；Quantized sampler 抛 TypeError；不支持类型抛 TypeError。
- `fixed_param_space` 为 Tune 固定变量路径→值。

### `_create_bohb_search`（第 153～170 行）

- 返回 `(TuneBOHB(space=cs, metric, mode=max), fixed_param_space)`。

### `_create_nevergrad_search`（第 173～190 行）

- `NevergradSearch(OnePlusOne, metric, mode=max)`。

### `_convert_zoopt_search_space`（第 193～215 行）

- flatten、fixed 参数；`ZOOptSearch.convert_search_space` 得 dim_dict。

### `_create_zoopt_search`（第 218～238 行）

- `ZOOptSearch(algo=asracos, budget=iterations, dim_dict, metric, mode=max)` 与 fixed 空间。

### `_resolve_ray_search_alg`（第 241～323 行）

- **第 261～262 行**：search_alg None 则 `(None, space, None)`。
- **第 264～272 行**：字符串空或 random 返回 None searcher；未知字符串 ValueError。
- **第 275～295 行**：ax/bohb/nevergrad/zoopt 分支创建或复用传入对象。
- **第 296～319 行**：非字符串对象直接返回；否则 check_requirements；optuna 用 multivariate TPE `OptunaSearch`；其它 `create_searcher`。
- **第 320～323 行**：ImportError 包装为 ModuleNotFoundError。

### `run_ray_tune`（第 326～506 行）— 依赖与默认空间

- **第 362～371 行**：`check_requirements` ray>=2.41 与 tune、pydantic 约束。
- **第 373～401 行**：`default_space` 大量 `tune.uniform/randint` 覆盖 lr、增强、loss 增益等。

### `run_ray_tune` — trial 与数据

- **第 404～406 行**：`task`、`ray.put(model)`、`base_name`。
- **第 408～431 行**：`_tune(config)`：ray.get 模型、清 trainer、合并 config 与 train_args；试取 trial_id 设唯一 `name`；`train`**；多 data 列表时对 metric 求平均返回 dict。
- **第 434～437 行**：无 space 用 default；非 resume 警告。
- **第 440～443 行**：注入 `data` 到 space 副本。

### `run_ray_tune` — 调度与 Tuner

- **第 445～447 行**：`_resolve_ray_search_alg(...)`。
- **第 450 行**：`tune.with_resources(_tune, cpu=NUM_THREADS, gpu=...)`。
- **第 453～469 行**：默认 `ASHAScheduler` grace/reduction；bohb 换 `HyperBandForBOHB`。
- **第 472～495 行**：`get_save_dir` 得 tune_dir；可 `Tuner.restore` 否则新建 `Tuner`（param_space、TuneConfig、RunConfig storage_path/name）。
- **第 498～506 行**：`fit()`、`get_results()`、`ray.shutdown()` 返回 results。