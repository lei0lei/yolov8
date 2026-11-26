# `tests/test_python.py`

Ultralytics 最核心的 Python 集成测试集：覆盖数据加载与 cfg 校验、设备选择、AutoBackend、checkpoint 安全加载、predict/val/train/track 全路径、Results 与标注、NDJSON/数据集工具、深度与语义专用逻辑、NN 模块 smoke、YOLOE/灰度/多光谱与大量 `utils` 行为。多数用例用小 `imgsz` 与 `coco8` 族数据；在线/慢测/边缘设备有条件 skip。

## 逐段说明

### `test_predict_prefetch`（第 49–60 行）

`_prefetch` 保序并在 loader 异常时原样 re-raise。

### `test_dataloader_caps_workers_to_batches`（第 63–75 行）

批次数过少时 `num_workers` 降为 0 或 ≤2。

### `test_dataloader_cap_preserves_distributed_drop_last`（第 78–94 行）

模拟 DDP rank2，`len(loader)==1`、`num_workers==0`、sampler seed 正确。

### `test_dataloader_seed_varies_sampling_order`（第 97–110 行）

相同 seed 可复现，不同 seed 顺序不同。

### `test_dataloader_empty_dataset_uses_dataloader_validation`（第 113–116 行）

空列表 `build_dataloader` 抛 `ValueError`。

### `test_image_cache_shared_with_spawned_workers`（第 119–129 行）

spawn worker 共享 `_ImageCache` 缓冲且内容一致。

### `test_build_yolo_dataset_hyp_isolated`（第 132–137 行）

`build_yolo_dataset` 不改共享 cfg 上 `mosaic` 默认值。

### `test_cfg_rejects_fuzzed_values`（第 140–173 行）

多组非法 `get_cfg`/`fraction`/`end2end` 映射；NumPy float64 yaml 加载正确。

### `skip_rpi_semantic`（第 176–179 行）

树莓派跳过语义测试的辅助函数。

### `test_select_device`（第 182–241 行）

**第一段**：mock CUDA，默认 `''` 不 `set_device`，`"1"` 设设备且不写 `CUDA_VISIBLE_DEVICES`。**第二段**：`parse_device` 与 `CUDA_VISIBLE_DEVICES` 映射、`-1` 空闲 GPU、畸形 CVD 前缀。

### `test_autobackend_memory_format`（第 244–271 行）

CPU channels_last 与 channels_last 权重 reload；inference_mode 下行为。

### `test_restricted_load_threaded`（第 274–284 行）

32 线程并发 `torch_safe_load` 不破坏 `pathlib.WindowsPath`。

### `test_temporary_modules_restores_preexisting_alias`（第 287–306 行）

`temporary_modules` 恢复已存在模块别名。

### `test_restricted_load_criterion`（第 309–329 行）

参数化 fused；legacy pt 含 criterion 与 `best_fitness` 安全加载前后向一致。

### `test_model_forward`（第 332–341 行）

多 yaml 前向；YOLO26 SPPF 无 SiLU；`augment=True` 无源。

### `test_tiled_prediction_single_window`（第 344–355 行）

单 tile 全图与不分 tile 框一致；异形图 tile 窗口列表。

### `test_tiled_merge_drops_seam_fragments`（第 358–372 行）

`merge_tiles` 丢弃接缝碎片框。

### `test_tiled_dataset_targets`（第 375–393 行）

`tile` 数据集裁切与标签；`load_image` 无 base；`full=True` 保留多边形。

### `test_model_methods`（第 396–413 行）

info/reset/load/fuse/callback/names/device/transforms/task_map。

### `test_model_load_remaps_cls_head_by_names`（第 416–438 行）

检测/分割/语义 head bias 按类名重排；YOLOE embedding 头不重排。

### `test_model_profile`（第 441–447 行）

`DetectionModel.predict(..., profile=True)` 不抛错。

### `test_predict_txt` / `test_predict_csv_*`（第 450–477 行）

多行 txt/csv 源共 7 条结果。

### `test_predict_img`（第 480–501 行）

`MODELS` 各权重：PIL/ndarray/tensor/流式/多源 batch。

### `test_preprocess_values`（第 504–513 行）

RGB/BGR 与灰度通道预处理与 `torch.equal` 首像素。

### `test_predict_classes_with_max_det`（第 516–528 行）

`classes`+`max_det` 先于截断；下次 predict 不泄漏 kwargs。

### `test_predict_visualize`（第 531–536 行）

`MODELS` + `visualize=True` smoke。

### `test_load_tensor_uint8`（第 539–547 行）

`LoadTensor` uint8→float；float eps 不截断。

### `test_predict_gray_and_4ch`（第 550–575 行）

灰度/RGBA/unicode 路径/EXIF/空格文件名 predict。

### `test_imread_exif_orientation`（第 578–589 行）

`imread` 多格式 EXIF 方向校正尺寸。

### `test_predict_ndarray_channels`（第 592–604 行）

2D 灰度扩展；`LoadPilAndNumpy` 通道变换无 float64 泄漏。

### `test_single_check_channel_order_and_contiguity`（第 607–623 行）

BGR/RGBA/灰度/16bit `_single_check` 像素与 C 连续。

### `test_predict_all_image_formats`（第 626–649 行，`slow`+ONLINE）

coco12-formats 十二种扩展名各一结果。

### `test_youtube`（第 652–662 行）

YouTube predict 或捕获网络错误。

### `test_track_second_association_indices`（第 665–678 行）

ByteTrack 二次关联保留原 detection 索引。

### `test_track_split_detections_degenerate_boxes`（第 681–701 行）

零/负尺寸框不进入 high/low 分区。

### `test_track_second_association_low_conf_keeps_id`（第 704–721 行）

bytetrack/fasttrack 低分第二关联保持 ID。

### `test_tracktrack_new_lifecycle`（第 724–753 行）

`min_track_len` 确认轨迹；首帧不自动激活行为。

### `test_track_reid_auto_user_detections`（第 756–769 行）

ReID `model=auto` 在用户检测下不崩溃。

### `test_track_stream`（第 772–807 行）

全 tracker 与 BoT-SORT GMC/ReID/onnx 变体在线视频。

### `test_val`（第 810–828 行）

各任务 metrics 导出与混淆矩阵维度（depth 无 cm）。

### `test_val_save_txt_pose`（第 831–854 行）

pose txt/json 关键点在 [0,1] 且与框对齐。

### `test_val_save_json_semantic`（第 857–878 行）

语义 save_json 与原图尺寸 mask 像素精度接近 metrics。

### `test_pose_metrics_curves`（第 881–886 行）

8 条唯一 pose 曲线名。

### `test_train_multi`（第 889–901 行）

双 coco8 MultiTrainer 目录与 json/png。

### `test_normalize_platform_uri`（第 904–911 行）

Platform URL→`ul://` 重写。

### `test_convert_signed_ndjson` / `test_ndjson_conversion_concurrency_and_resume`（第 914–1060 行）

异步转换挂钩；detect/classify 并发与 resume 下载计数。

### `test_train_scratch` / `test_train_ndjson` / `test_train_pretrained`（第 1063–1092 行）

scratch/ndjson/pretrained 短训 smoke。

### `test_all_model_yamls`（第 1095–1102 行）

`cfg/models` 下 yaml 可构建（rtdetr 需 torch1.11）。

### `test_workflow`（第 1105–1112 行）

train→val→predict→export torchscript。

### `test_predict_callback_and_setup`（第 1115–1135 行）

`on_predict_batch_end` 与预构建 `load_inference_source`。

### `test_results`（第 1138–1164 行）

各模型 Results 导出 txt/crop/df/csv/json/plot。

### `test_results_plot_without_boxes`（第 1167–1176 行）

仅 masks 时 `plot` 两种 color_mode。

### `test_results_plot_instance_keypoints_track_color`（第 1179–1191 行）

`color_mode=instance` 关键点色跟 track id。

### `test_results_depth_*` / `test_annotator_depth_map`（第 1194–1242 行）

DepthMap 链式转换、plot 叠加、全零 depth 不除零。

### `test_colors_parse`（第 1245–1251 行）

非法颜色串返回 None；合法十六进制解析。

### `test_dense_result_tensor_indexing` / `test_results_plot_empty_dense_selection`（第 1254–1286 行）

语义/深度稠密索引与空切片 plot。

### `test_annotator_tensor_image` / `test_results_update_probs`（第 1289–1312 行）

张量输入 Annotator；`update(probs=)` 包装 `Probs`。

### `test_labels_and_crops`（第 1315–1345 行）

save_txt/crop 数量与目录与类名一致。

### `test_data_utils`（第 1348–1387 行）

autosplit、zip 清元数据、数据集 split 错误、`polygons2masks_overlap` 130 实例不溢出。

### `test_safe_download_*`（第 1391–1457 行）

本地 zip/tar/tgz 解压路径；zip slip 与 tar slip 跳过。

### `test_data_converter` / `test_data_annotator`（第 1460–1485 行）

COCO→YOLO 与 auto_annotate smoke（ONLINE）。

### `test_events`（第 1488–1496 行）

Events 发送 mode=test。

### `test_cfg_init`（第 1499–1549 行）

`smart_value` 与注入防护；zipfile 常量解析。

### `test_depth_calibration_checkpoint_provenance` / `test_depth_trainer_records_portable_calibration_split`（第 1552–1619 行）

深度校准元数据写入 ckpt；`final_eval` 记录 validation_split。

### `test_verify_image_label_keeps_polygons_sharing_a_box`（第 1622–1631 行）

同框两 polygon 均保留。

### `test_depth_dataset_ignores_unreadable_targets`（第 1634–1665 行）

坏深度图跳过；cache 生成；hyp 不被污染。

### `test_utils_init` / `test_utils_checks` / `test_utils_benchmarks` / `test_utils_torchutils`（第 1668–1731 行）

ubuntu 版本；`parse_version`/`check_imgsz`；ProfileModels；`profile_ops`。

### `test_rmsprop_checkpoint_preserves_square_avg`（第 1734–1761 行）

fp16 optimizer state 与 resume 更新幅度一致。

### `test_rtdetr_remap_cls_by_names`（第 1764–1791 行）

RT-DETR score/denoising embedding 按名重排。

### `test_semantic_loss_all_ignore` / `test_semantic_confusion_matrix_large_counts`（第 1794–1820 行）

全 ignore 有限 loss；大计数不 float32 饱和。

### `_DepthLossModel` / `_depth_loss_for_scaled_pred` / `test_v26_depth_loss_lower_lambda_penalizes_scale_error_more`（第 1823–1854 行）

深度 SILog λ 对尺度误差灵敏度。

### `test_utils_ops` / `test_scale_*` / `test_nms_*` / `test_process_mask_empty`（第 1857–1973 行）

坐标变换往返；letterbox 反缩放；e2e NMS 类过滤；空 mask。

### `test_utils_files` / `test_utils_patches_torch_save`（第 1976–2010 行）

路径工具；`torch_save` 失败重试 4 次。

### `test_nn_modules_conv` / `test_nn_modules_block`（第 2013–2044 行）

未覆盖 conv/block 前向与 Conv2 fuse。

### `test_nn_detect_head_export_clamps_max_det`（第 2047–2055 行）

导出 postprocess 不超 anchor 数。

### `test_nn_aifi_pos_embed_row_major`（第 2058–2065 行）

AIFI 位置编码行列分解结构。

### `_depth_head_feats` / `test_nn_depth_head_*`（第 2068–2094 行）

导出 4× 上采样；全参数有梯度。

### `test_classification_fraction_samples_across_classes` / `test_classification_split_class_alignment`（第 2097–2138 行）

分类 fraction 分层；split 与 model names 对齐丢类。

### `image` fixture（第 2141–2143 行）

提供 BGR `SOURCE` 图像。

### `test_classify_transforms_train`（第 2156–2180 行）

参数化 auto_augment/erasing；输出 224 张量。

### `test_model_tune` / `test_model_tune_ray`（第 2183–2210 行）

本地与 Ray tune 短跑。

### `test_model_embeddings`（第 2213–2227 行）

detect/seg embed 与 classify embed 层索引。

### `test_process_mask_local` / `test_process_mask_native_chunked`（第 2230–2263 行）

局部 mask 与 dense 一致；分块 native 等同一次性。

### `test_yoloe` / `test_yoloe_vocab_head_switch` / `test_yoloe_visual_prompt_verbose_false`（第 2272–2397 行）

文本/视觉提示、训练器变体、vocab 切换、`verbose=False` 无 Ultralytics 日志。

### `test_yolov10` / `test_multichannel` / `test_grayscale`（第 2400–2456 行）

v10 全流程；多光谱 TIFF；各任务灰度 yaml 训 val 导出。

### `test_semantic_polygon_data` / `test_semantic_cache_nc_edit_1bit_masks`（第 2459–2482 行）

polygon 语义训练；1-bit mask nc 改动后仍 {0,1}。

### `test_verify_image_label_whitespace_lines` / `test_load_yolo_dota_whitespace_lines`（第 2485–2515 行）

标签空白行不标 corrupt；DOTA 加载跳过空行。