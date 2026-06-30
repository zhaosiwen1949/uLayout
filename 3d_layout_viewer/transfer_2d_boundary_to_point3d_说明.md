# transfer_2d_boundary_to_point3d 说明

把布局推理得到的 2D 边界（`boundary`）对应到 COLMAP SfM 的真实 3D 点，汇成一个带颜色的场景
点云。对应脚本 `3d_layout_viewer/transfer_2d_boundary_to_point3d.py`。

## 1. 作用

`output/<scene>_uLayout/inference_img/panorama_pred_boundary/est_*.json` 里每帧记录了
`boundary`（shape `[2, 1024]`，弧度，每列一个值）：

- `boundary[0]`：天花线 vc，范围 −π/2 ~ 0（朝上）；
- `boundary[1]`：地面线 vf，范围 0 ~ π/2（朝下）。

脚本把每个边界点从**全景坐标系**投到 rig 各**子相机的透视像素**，与 `images.bin` 里记录的
2D 特征点比像素距离；命中（阈值内）后用该特征点的 `point3D_id`，经 `points3D.bin` 取得 3D
坐标与颜色，所有命中点**按 point3D_id 去重合并**成一个 `.ply`。

> 不是每一列都需要命中：边界线上没有 SfM 特征点的列会被跳过。

## 2. 数据环境（COLMAP 标准输出）

固定读取 `<dataset_dir>/sparse/0/`：

| 文件 | 用途 |
|---|---|
| `cameras.bin` | 6 个 PINHOLE 子相机内参 `(fx, fy, cx, cy)`，按 `cam_id` |
| `images.bin` | 每张子图 `pano_cameraN/frame_XXXXX.jpg` 的 `cam_id` 与 2D 特征点 `(x, y, point3D_id)`；`cam_id = N+1`，`frame_key = frame_XXXXX` 与 est 的 `img_name` 一致 |
| `points3D.bin` | `point3D_id → (xyz, rgb)` |
| `rigs.bin` | 各子相机相对参考相机 `pano_camera0` 的 `sensor_from_rig` 旋转 |

`images.bin` 较大（GB 级），首次解析后把**有效**特征点（`point3D_id != -1`）缓存到
`sparse/0/feat2d3d_cache.npz`，后续秒级加载；`--rebuild-cache` 可强制重建。该缓存即“记录
images.bin 中 2D→3D 关系”的落地（在脚本内完成，不改 `CustomPanoDataset`）。

## 3. 坐标变换链（核心）

复用 `multi_layout_viewer` 的 `GEOS_TO_PANO`、`reference_rig_rotation`，以及
`layout_3d_utils.np_coorx2u`。轴向约定见 `坐标变换说明.md`：

```
geos(right=+X, up=+Z, front=−Y)
  --GEOS_TO_PANO-->  pano(right=+X, up=+Y, front=+Z)
  --Ror.T-->         pano_camera0 相机(right=+X, down=+Y, forward=+Z)   # Ror=Rx(AX)·Ry(AY): cam0→pano，取逆
  --R_rig[X]-->      pano_cameraX 相机                                  # rigs.bin sensor_from_rig，cam0=单位
```

对每个边界点（列 `c`、弧度 `v`，天花 v<0 / 地面 v>0）：

```python
u      = np_coorx2u(c, 1024)
d_geos = [cos(v)·sin(u), −cos(v)·cos(u), −sin(v)]   # 单位 bearing，与 get_3d_layout 一致
d_cam0 = Ror.T @ (GEOS_TO_PANO @ d_geos)
for X in 0..5:                                       # cam_id = X+1
    d = R_rig[X] @ d_cam0
    if d[2] <= 0: continue                           # 在相机背后
    px = fx·d[0]/d[2] + cx ; py = fy·d[1]/d[2] + cy  # PINHOLE 投影
    if (px,py) 不在 [0,W)×[0,H): continue
    在该帧子图 X 的特征点里取最近 (x,y)，距离 < pixel_thresh 则命中
取所有命中子相机里最近的那个 → point3D_id → points3D[id]
```

### 近似与注意

- **忽略 rig 的 cm 级平移**：边界是无深度的方向射线，按纯旋转投影各子相机；近墙处会带来约
  几个像素的视差误差，由 `--pixel-thresh` 兜底。
- **全景↔rig 朝向**：`Ror=Rx(35)·Ry(−40)` 是参考标定（见 `坐标变换说明.md`），可用
  `--rig-angles` 调。若命中率异常低，多半是该朝向不准——加大阈值或微调角度。
- **逐帧只有特征点不同**：boundary→子相机像素映射只依赖固定的全景↔rig 朝向，对所有帧相同；
  每帧只换各子图的特征点集合。

## 4. 用法

```bash
cd 3d_layout_viewer
# 默认：dataset_dir=../src/xinghecheng，layout=../output/xinghecheng_uLayout/.../panorama_pred_boundary
python transfer_2d_boundary_to_point3d.py --pixel-thresh 5

# 指定其它数据集 / 子集 / 阈值
python transfer_2d_boundary_to_point3d.py \
    --dataset_dir ../src/foo \
    --layout ../output/foo_uLayout/inference_img/panorama_pred_boundary \
    --curves both --pixel-thresh 8 --indices 0 1 2
```

输出：`output/<scene>_uLayout/inference_ply/boundary_points3d.ply`（带颜色，二进制 PLY；
`scene` 取 `--dataset_dir` 目录名）。

## 5. 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset_dir` | `../src/xinghecheng` | 含 `sparse/0` 与 `img` 的数据集目录 |
| `--layout` | `../output/xinghecheng_uLayout/.../panorama_pred_boundary` | est_*.json 边界目录 |
| `--rig-angles AX AY` | `35 -40` | `Ror = Rx(AX)·Ry(AY)`：pano_camera0 → 全景 |
| `--pixel-thresh` | `5` | 匹配阈值（子相机像素） |
| `--curves` | `both` | 转哪条边界：`floor` / `ceiling` / `both` |
| `--indices` | 全部 | 只处理部分 est 索引（调试用） |
| `--out` | 自动 | 覆盖默认 PLY 输出路径 |
| `--rebuild-cache` | 关 | 强制重建 `feat2d3d_cache.npz` |

## 6. 输出与校验

- 终端打印：相机/rig/points3D 数量、扫描帧数与边界列数、最终唯一 3D 点数。
- 命中点数随 `--pixel-thresh` 单调增长且小阈值即有命中，说明朝向链准确（若大阈值才突然出现
  命中，则全景↔rig 朝向有系统偏差，需调 `--rig-angles`）。
- 用 open3d / MeshLab 打开生成的 PLY，应能看到带颜色、贴合房间墙体轮廓的点云，可与 SfM 稠密
  点云 `sparse/0/points3D.ply` 套合对照。
