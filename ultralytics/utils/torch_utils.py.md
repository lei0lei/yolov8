# `ultralytics/utils/torch_utils.py`

本文件是 Ultralytics 训练与推理的 PyTorch 工具层：在导入时建立 `TORCH_*`/`TORCHVISION_*` 版本布尔常量，并提供设备解析与选择、分布式屏障、AMP `autocast`、卷积-BN 融合、QAT（ModelOpt）、EMA、早停、FLOPs/显存剖析、确定性训练与 `torch.compile` 封装。`parse_device` 与 `select_device` 是 CLI 与 Trainer 指定 GPU/CPU/MPS/NPU/XPU 的单一入口，且尊重外部 `CUDA_VISIBLE_DEVICES` 而不改写该环境变量。

## 逐段说明

### 模块级常量（第 44～67 行）

对各 PyTorch/torchvision 版本调用 `check_version` 得 `TORCH_1_9`…`TORCH_2_13` 等。Windows 上 torch==2.4.0 时 `LOGGER.warning` 建议升级。

### `get_torch_device_backend`（第 70～73 行）

从 `device.type` 或字符串解析设备类型；有 `torch.get_device_module` 则用之，否则 `getattr(torch, device_type)`。

### `torch_distributed_zero_first`（第 76～86 行）

上下文管理器：已初始化分布式且 `local_rank` 非 0/-1 时先 `barrier`（NCCL 带 `device_ids`）；`yield` 后 rank 0 再 `barrier`，保证 rank0 先完成准备。

### `smart_inference_mode`（第 89～109 行）

返回装饰器工厂。`mode=False`：≥1.9 用 `inference_mode(False)+no_grad`，否则仅 `no_grad`。`mode=True`：已在 inference_mode 则透传；≥1.10 用 `inference_mode` 否则 `no_grad`。

### `autocast`（第 112～160 行）

`enabled` 可为 `torch.dtype` 指定 dtype。`bfloat16` 时校验 CUDA 与 torch≥1.13 及本机 bf16 支持，否则 `RuntimeError`。`device=="npu"` 返回 `torch_npu.npu.amp.autocast`。≥1.10 用 `torch.autocast`；MPS 且 <2.5 降级 cpu 并关 AMP。旧版回退 `torch.cuda.amp.autocast`。

### `get_cpu_info` / `get_gpu_info`（第 163～173 行）

`lru_cache`；CPU 名来自 `CPUInfo.name()`；GPU 为名称与 MiB 显存字符串。

### `parse_device`（第 176～241 行）

`torch.device` 无 index 的 cuda 返回 `''`；npu/xpu 格式化为 `type` 或 `type:idx`。清洗字符串前缀后缀；`cuda`→`0`；`cpu:N`→`cpu`。解析 npu/xpu 逗号索引。数字索引与 `CUDA_VISIBLE_DEVICES` 物理 id 互译。`-1` 占位由 `GPUInfo().select_idle_gpu` 逐个替换为空则剔除。

### `select_device`（第 244～369 行）

非 cpu/cuda/npu/xpu 的 `torch.device` 或 tpu/vulkan 字符串原样返回。`parse_device` 后分支：NPU/XPU 校验包、后端可用性、索引合法并 `set_device`（单卡）；CUDA 非法则抛 `ValueError` 含环境诊断；空 device 且 CUDA 可用用当前或列表 GPU 并可选 `set_device`；MPS 条件满足则选 mps；否则 CPU 并 `set_num_threads`。Mac ARM torch≥2.3 关 NNPACK。`verbose` 控制 `LOGGER.info`。

### `time_sync`（第 372～378 行）

非 cpu/mps 设备上对加速器 `synchronize` 后返回 `perf_counter`。

### `fuse_conv_and_bn`（第 381～403 行）

`fuse_conv_bn_weights` 折叠 BN 到 conv 权重与偏置；存在 `weight_quantizer` 时同步缩放 `amax`；`requires_grad_(False)`。

### `fuse_deconv_and_bn`（第 405～430 行）

`bn` 为 `Identity` 则仅冻结 deconv。重排 transpose 权重布局后同样 fuse；写回 `nn.Parameter` 形状的转置权重。

### QAT 系列（第 438～541 行）

`prepare_qat`：rank0 先 `check_requirements(MODELOPT_REQUIREMENTS)`，eval+`mtq.quantize` 校准若干 batch，对 head 输出层 `disable_quantizer`，再恢复 train 模式。`is_qat` 按模块类名 `TensorQuantizer` 判断。`qat_state` 抽取 modelopt state 与 quantizer 张量；蒸馏用 `student_model`。`strip_qat` 导出 DynamicModule 并清 runtime state。`restore_qat` 加载 state 与 ranges。

### `model_info`（第 544～586 行）

`verbose=False` 直接返回。统计参数量、梯度数、叶子层数；`detailed` 时逐参数打印统计；`get_flops` 后打印摘要，返回 `(n_l,n_p,n_g,flops)`。

### `get_num_params` / `get_num_gradients`（第 589～596 行）

对 `parameters()` 的 `numel` 求和，后者过滤 `requires_grad`。

### `model_info_for_loggers`（第 599～629 行）

`profile` 为真时 `ProfileModels` 测 ONNX/TRT；否则仅参数量与 GFLOPs；始终写入最近一次验证的 PyTorch 推理毫秒。

### `get_flops`（第 632～669 行）

无 thop 返回 0。`unwrap_model` 后按 stride 与 RT-DETR `min_cells` 构造假输入，`thop.profile`×2 转 GFLOPs；异常返回 0。

### `initialize_weights`（第 672～680 行）

遍历模块：Conv2d 保持默认初始化；BatchNorm2d 设 `eps=1e-3`、`momentum=0.03`。

### `scale_img`（第 683～702 行）

`ratio==1` 原样返回。双线性缩放；`same_shape` 为假则 pad 到 `gs` 倍数，填充值 ImageNet mean 0.447。

### `copy_attr` / `intersect_dicts`（第 705～734 行）

按 include/exclude 从 `b.__dict__` 复制到 `a`，跳过 `_` 前缀。字典交集且 shape 相同。

### `is_parallel` / `unwrap_model`（第 737～765 行）

判断 DP/DDP；循环剥离 `_orig_mod` 与 `.module`。

### `one_cycle`（第 768～779 行）

返回余弦 ramp lambda，从 y1 到 y2。

### `init_seeds`（第 782～804 行）

设置 random/numpy/torch 种子。`deterministic` 为真且 torch≥2.0 启用确定性算法、cudnn、环境变量；否则调用 `unset_deterministic`。

### `unset_deterministic`（第 807～814 行）

关闭确定性算法并弹出相关环境变量。

### `interpolate_bilinear`（第 817～837 行）

CPU 或未开确定性或未达分解条件时用原生 `F.interpolate`。否则用 eye 矩阵经 1D 线性核构造权重，双矩阵乘实现双线性（规避确定性下原生 kernel 分解）。

### 类 `ModelEMA`（第 840～915 行）

`__init__`：`deepcopy(unwrap_model).eval()`，蒸馏时清空 `teacher_model`；`decay` 为 tau 控制的指数 ramp；冻结 EMA 参数；`_pairs` 延迟构建。`update`：`enabled` 时递增 `updates`，算 `d`，首次收集浮点 state 对，≥2.0 且非 npu（mps 需≥2.4）用 `torch._foreach_lerp_`，否则逐张量 `mul_`+`add_`。`update_attr`：`copy_attr` 同步非参数属性。

### `strip_optimizer`（第 918～983 行）

`torch_load` 校验 dict 含 model；有 ema 则替换 model；蒸馏 unwrap student；`args` 转 dict，清 criterion；half 并冻结参数；合并 `DEFAULT_CFG` 与 `train_args` 写回；清理 optimizer/ema 等键；`torch.save` 元数据与权重。

### `convert_optimizer_state_dict_to_fp16`（第 986～1007 行）

遍历 optimizer state 张量，step/exp_avg 等保持 fp32，其余 fp32 转 half。

### `cuda_memory_usage`（第 1011～1036 行）

cpu/mps 仅 yield 零字典。否则 `empty_cache`，finally 写入 `memory_reserved` 字节数。

### `profile_ops`（第 1039～1117 行）

`select_device` 后循环 input×ops：清缓存、同步计时 forward/backward、`thop` FLOPs、可选 AutoBatch 损失显存估算；打印表格行；异常记 None 并继续 gc/empty_cache。

### 类 `EarlyStopping`（第 1120～1169 行）

`patience` 为 0/None 变 `inf`。`__call__`：`fitness is None` 不停止；刷新 best 或允许初始零；`delta` 超 patience 打日志并返回 True。

### `attempt_compile`（第 1172～1271 行）

无 compile 或 `mode` 假/QAT 模型返回原模块或抛错。字符串 mode 默认 default。预检 CPU 编译器与 CUDA Triton；失败警告并返回原模型。`torch.compile(inductor)`；可选 warmup 在 autocast/inference_mode 下 dummy 前向并 CUDA sync；记录 compile+warmup 耗时。