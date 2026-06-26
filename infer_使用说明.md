# infer 使用说明

记录 `infer.py` 的使用方式（后续 infer 相关信息统一更新到本文件）。

## 数据目录结构

推理数据放在 `data_dir` 指定的目录下，**不再有 mode 一级子目录**，全景图直接放在
`<data_dir>/img`：

```
<data_dir>/
├── img/                # 待推理的全景图（.jpg / .png）
├── images.bin          # （可选）COLMAP 相机外参，供 multi_layout_viewer 使用
└── camera_poses.json   # （可选）由 images.bin 自动生成的缓存
```

> 不读取、也不需要 `label_cor`（groundTruth）。自定义数据集没有真值。

## 运行

数据路径通过脚本参数 `data_dir` 指定（hydra 命令行覆盖，形式为 `key=value`），与
`mode=`、`ckpt_dir=`、`save_pred=` 等参数用法一致。

```bash
# 用默认路径（src/custom）
python infer.py mode=test ckpt_dir=ckpt/best_mp3d.pth save_pred=true

# 用脚本参数指定其它数据路径
python infer.py data_dir=/path/to/my_data mode=test ckpt_dir=ckpt/best_mp3d.pth save_pred=true
```

- 不传 `data_dir=` 时用默认值 `src/custom`，此时全景图需放在 `src/custom/img`。
- 传 `data_dir=<路径>` 即可切换推理数据，无需修改配置文件。

### 关于 `mode`

`mode` 默认值已是 `test`，**可省略 `mode=test`**，行为不变。`mode` 现在只影响：

1. **输出目录名**：`id_exp = "ulayout_<pano>_<pp>_<mode>"`，`test` → 输出到
   `output/ulayout_mp3d_lsun_test/...`。viewer 脚本（`view.sh` / `multi_view.sh`）里写死
   引用的就是 `ulayout_mp3d_lsun_test`，保持 `test` 才能对上。
2. **数据集增强开关**：`mode=='train'` 会打开 flip/rotate/gamma/stretch，但改动后
   `__getitem__` 只返回图片、忽略这些开关，所以**实际无影响**。
3. `mode` **不再影响数据路径**（已去掉 mode 子目录）。

> 建议保持 `test`（或不传）。若传 `mode=train`，输出目录会变成
> `ulayout_mp3d_lsun_train`，与 viewer 脚本写死的路径对不上。

## 输出

infer 调用 `plot_pano_custom`（针对无真值的自定义数据集，仅处理预测结果）：

```
output/<id_exp>/inference_img/
├── panorama/                   # 预测边界叠加在原图上的可视化 est_<i>_<name>.jpg
└── panorama_pred_boundary/     # save_pred=true 时输出，est_<i>.json（仅含预测 boundary）
```

- `panorama_pred_boundary/est_<i>.json` 字段：`img_name`、`boundary`（预测）、`corner`（空，
  因无真值）。该 json 供 `3d_layout_viewer` 渲染 3D / 2D 户型图使用。
