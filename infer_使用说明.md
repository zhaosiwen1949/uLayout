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
`ckpt_dir=`、`save_pred=` 等参数用法一致。

```bash
# 用默认路径（src/custom）
python infer.py ckpt_dir=ckpt/best_mp3d.pth save_pred=true

# 用脚本参数指定其它数据路径
python infer.py data_dir=/path/to/my_data ckpt_dir=ckpt/best_mp3d.pth save_pred=true
```

- 不传 `data_dir=` 时用默认值 `src/custom`，此时全景图需放在 `src/custom/img`。
- 传 `data_dir=<路径>` 即可切换推理数据，无需修改配置文件。

## 输出

输出根目录为 **`output/<data_dir 目录名>_uLayout`**：取 `data_dir` 的目录名加后缀
`_uLayout`。例如 `data_dir=src/xinghecheng` → `output/xinghecheng_uLayout`；默认
`data_dir=src/custom` → `output/custom_uLayout`。

infer 调用 `plot_pano_custom`（针对无真值的自定义数据集，仅处理预测结果）：

```
output/<data_dir 目录名>_uLayout/inference_img/
├── panorama/                   # 预测边界叠加在原图上的可视化 est_<i>_<name>.jpg
└── panorama_pred_boundary/     # save_pred=true 时输出，est_<i>.json（仅含预测 boundary）
```

- `panorama_pred_boundary/est_<i>.json` 字段：`img_name`、`boundary`（预测）、`corner`（空，
  因无真值）。该 json 供 `3d_layout_viewer` 渲染 3D / 2D 户型图使用。

## 换数据集时保持名字一致

输出目录名由 `data_dir` 推导（`output/<data_dir 目录名>_uLayout`），而 viewer 脚本里的
`--dataset_dir` 和 `--layout` 是写死的。换数据集（例如从 `xinghecheng` 换成 `foo`，数据放在
`src/foo/img`）时，以下 **3 处的名字必须保持一致**，否则 viewer 会因 `img_name` 对不上而
渲染错误（布局与相机位姿来自不同数据集）：

| 位置 | 字段 | 以 `src/foo` 为例 |
|---|---|---|
| 运行 infer | `data_dir=` | `data_dir=src/foo` → 输出 `output/foo_uLayout/...` |
| `view.sh` / `multi_view.sh` | `--dataset_dir` | `../src/foo` |
| `view.sh` / `multi_view.sh` | `--layout` | `../output/foo_uLayout/inference_img/panorama_pred_boundary` |

要点：

- `--dataset_dir` 决定相机外参（`images.bin`）来源，`--layout` 决定布局 json 来源；二者
  必须指向**同一数据集**，名字才能对上。
- `output/<data_dir 目录名>_uLayout` 里的 `<data_dir 目录名>` 就是 `data_dir` 的最后一级
  目录名，所以三处用同一个数据集名即可串起来。
- 当前脚本已配置为 `xinghecheng`（`--dataset_dir ../src/xinghecheng`、
  `--layout ../output/xinghecheng_uLayout/...`）。
