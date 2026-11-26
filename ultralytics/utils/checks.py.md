# `ultralytics/utils/checks.py`

`checks.py` 在包加载与环境自检阶段承担依赖解析、版本比对、资源路径解析与自动安装，是 `yolo checks`、训练前 AMP 探测以及 `check_file`/`check_requirements` 等路径的集中实现。模块顶部从 `ultralytics.utils` 引入平台标志、`AUTOINSTALL`、`LOGGER` 等；末尾在导入时执行 Python/torchvision 地板检查并导出 ONNX Runtime 相关常量。Platform `ul://` URI 经 API 解析为签名 HTTPS URL，与远程权重/数据集下载协同。

## 逐段说明

### `normalize_platform_uri`（第 61～71 行）

若字符串以 `PLATFORM_URL/` 开头，改写为 `ul://` 加路径段；否则原样返回。

### `resolve_platform_uri`（第 74～144 行）

解析 `ul://user/datasets|project/...` 三段式 URI；无 `ULTRALYTICS_API_KEY` 或 settings 中 key 则 `ValueError`。懒导入 `Platform`+`httpx`，数据集用长超时 `export`，模型取 `files[0].downloadUrl`。`APIConnectionError`：hard 抛 `ConnectionError` 否则 warning+None。`APIError` 按 401/403/408/429/5xx/404 分支抛错或 warning。hard 模式下 404 抛 `FileNotFoundError`。

### `parse_requirements`（第 147～184 行）

`package` 非空时从 distribution metadata 按 extra 与 marker 过滤；否则读 requirements.txt 行。去注释后正则解析 name+specifier 为 `SimpleNamespace` 列表。

### `get_distribution_name`（第 187～193 行）

遍历 distributions 的 `top_level.txt` 匹配 import 名，找不到则返回原名。

### `parse_version`（第 197～215 行）

`lru_cache`；正则提取数字段转 tuple，不足三位补 0；失败 warning 并 (0,0,0)。

### `is_ascii`（第 218～227 行）

`str(s).isascii()`。

### `check_imgsz`（第 230～294 行）

stride 张量取 max 转 int。int→单元素 list；str 数字或 `literal_eval` 列表。超 `max_dim`：`max_dim!=1` 抛错否则 warning 取 max。每维 `ceil` 到 stride 倍数且 ≥floor/stride。与输入不同 warning。`min_dim==2` 且一维则复制成方阵。

### `check_uv`（第 298～303 行）

子进程 `uv -V` 返回码 0 为真。

### `check_version`（第 307～395 行）

current 空则 warning 真。current 非数字开头视作包名拉 metadata.version，未安装且 hard 抛 `ModuleNotFoundError`。required 含 `sys_platform` 且当前 OS 不匹配则真。逗号分隔多条件：解析 op 与版本，缺 op 默认 `>=`；扩展 tuple 长度比较 `== != >= <= > <`。不满足时 hard 抛错或 verbose warning。

### `check_latest_pypi_version` / `check_pip_update_available`（第 398～437 行）

请求 pypi json 取 version。在线且 pip 安装时比对 `__version__` 并 info 升级提示。

### `check_font`（第 443～472 行）

`ThreadingLocked`+`lru_cache`；先 `USER_CONFIG_DIR`，再 matplotlib ttflist，再 `ASSETS_URL` 下载。

### `check_python`（第 475～486 行）

委托 `check_version(PYTHON_VERSION, minimum, ...)`。

### `check_apt_requirements`（第 490～532 行）

`TryExcept`；`dpkg -l` 查缺失。有缺失且 `AUTOINSTALL`：sudo apt update/install，否则 warning。无 AUTOINSTALL 提示手动 apt。

### `check_requirements`（第 536～664 行）

`ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS` 直接 True。Path 用 `parse_requirements`+exclude。每条 requirement 可候选列表，任一满足 version+extra 子依赖则过；否则加入安装列表。`attempt_install`：`Retry` 包装 uv pip 或 python -m pip。需安装且 AUTOINSTALL+ONLINE 则安装并提示重启；否则 False。

### `check_executorch_requirements`（第 667～676 行）

ARM64 Docker 先 packaging≥22；torch<2.13 钉 `executorch<1.5` 并与当前 torch 版本同装。

### `check_tensorrt` / `check_torchvision` / `check_paddle`（第 679～727 行）

Linux 装 tensorrt 或 tensorrt-cuN。torchvision minor 与 torch major 不匹配 warning。paddle-gpu 存在则 version check，否则 CPU `paddlepaddle` 范围。

### `check_suffix` / `check_yolov5u_filename` / `check_model_file_from_stem`（第 730～788 行）

后缀集合断言；老 v5/v3 权重名替换为 u 版并 info；stem 在 `GITHUB_ASSETS_STEMS` 则补 `.pt`。

### `check_file`（第 791～850 行）

平台 URI 规范化→后缀→yolov5u 改名。本地存在、grpc、空路径直接返回。`ul://` 解析下载到 `download_dir/uri_path/`。HTTP(S)/gs 转 URL `safe_download`。否则 glob 仓库搜，hard 时唯一性断言。

### `check_yaml`（第 853～864 行）

`check_file` 包装 yaml 后缀。

### `check_data_portable`（第 867～880 行）

非 str/Path 或空为假；URL、无路径分隔符可移植名、或 `check_file` 找到为真。

### `check_is_path_safe`（第 883～896 行）

resolve 后前缀 parts 等于 basedir 且存在。

### `check_imshow`（第 900～921 行）

Linux 非 Colab/Kaggle 需 DISPLAY；试 `cv2.imshow` 8×8；失败可选 warning。

### `check_yolo` / `collect_system_info`（第 924～1017 行）

Colab 删 sample_data；verbose 时 psutil 内存磁盘；`check_multiple_install`；`select_device`；info 完成。`collect_system_info` 组装 OS/CPU/GPU 等 dict，遍历 ultralytics 包 requirements 打勾表，GHA 时附 GitHub 环境键。

### `check_amp`（第 1020～1086 行）

cpu/mps 直接 False。CUDA 上 GTX 16xx 等正则匹配则 warning+False。未训练 YOLO26n 注册 Conv/Attention hook 比较 FP32 与 autocast 输出范数比；全层通过才 True。

### `check_multiple_install`（第 1089～1114 行）

pip show 路径与当前 `ROOT` `samefile` 比较，冲突 warning 可编辑安装提示。

### `print_args`（第 1117～1140 行）

取上一帧参数，长 URL `clean_url`，相对 ROOT 打印彩色键值。

### `cuda_device_count` / `cuda_is_available` / `rocm_is_available`（第 1143～1167 行）

薄封装 torch.cuda 与 hip 平台判断。

### `is_rockchip` / `is_intel` / `is_sudo_available`（第 1170～1215 行）

device-tree compatible 与 RKNN_CHIPS；CPU 名或 xpu-smi；非 Windows 且 which sudo。

### 模块尾常量（第 1218～1232 行）

`check_python(3.8)`、`check_torchvision()`；`IS_PYTHON_*`、`ONNXRUNTIME_BUILDS`、`ONNXRUNTIME_CUDA` 拼接 CUDA12 上限。