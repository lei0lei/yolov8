# `ultralytics/solutions/object_cropper.py`

按预测框裁剪目标并保存到 `crop_dir`；强制 `tracker=None` 使用纯 predict，不支持 `show` 预览。

## 逐段说明

### `__init__`（第 30～45 行）

`kwargs["tracker"]=None`；创建 `crop_dir`；`show=True` 时警告并关闭；`crop_idx=0`。

### `process`（第 47～69 行）

`extract_tracks`；每框递增 `crop_idx`，`save_one_box` 到 `crop_{n}.jpg`；返回原图与 `total_crop_objects`。