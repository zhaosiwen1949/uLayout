# infer.py 使用方式

推理数据路径通过脚本参数 `data_dir` 指定（hydra 命令行覆盖，形式为 `key=value`），
与 `mode=`、`ckpt_dir=`、`save_pred=` 等参数用法一致。

```bash
# 用默认路径（src/custom）
python infer.py mode=test ckpt_dir=ckpt/best_mp3d.pth save_pred=true

# 用脚本参数指定其它数据路径
python infer.py data_dir=/path/to/my_data mode=test ckpt_dir=ckpt/best_mp3d.pth save_pred=true
```

- 不传 `data_dir=` 时用默认值 `src/custom`。
- 传 `data_dir=<路径>` 即可切换推理数据，无需修改配置文件。
