# resize_panorama.py 说明

把全景图批量缩放到 **1024×512**（等距柱状 equirectangular 分辨率），供 uLayout
推理（`infer.py`）使用。

## 功能

- 读取输入文件夹下的所有图片（`.jpg/.jpeg/.png/.bmp`），逐张缩放为 `1024×512`
  （宽×高，BICUBIC 插值，转为 RGB），按原文件名保存到输出文件夹。
- 输出文件夹不存在时自动创建。

## 用法

```bash
# 指定输入目录，输出到默认位置（输入目录上一级的 img 文件夹）
python tools/resize_panorama.py src/xinghecheng/panorama_images
# -> 输出到 src/xinghecheng/img

# 自定义输出目录
python tools/resize_panorama.py src/xinghecheng/panorama_images -o src/xinghecheng/img
```

## 参数

| 参数 | 说明 |
|---|---|
| `input_dir`（位置参数，必填） | 源图片所在文件夹 |
| `-o` / `--output`（可选） | 输出文件夹；默认 `<input_dir>/../img`，即输入目录**上一级**的 `img` 文件夹 |

## 输出分辨率

固定为 `1024×512`（脚本内常量 `TARGET_W=1024, TARGET_H=512`）。

## 备注

- 与 `infer.py` 的数据目录约定衔接：默认输出到 `<数据根目录>/img`，正是推理时
  `data_dir` 所指目录下读取图片的位置（见 `infer_使用说明.md`）。
- 需要 `Pillow`（PIL）。在项目 conda 环境 `uLayout` 下运行。
