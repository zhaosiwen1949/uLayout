#!/usr/bin/env bash
# python infer.py mode=test ckpt_dir=ckpt/best_mp3d.pth save_pred=true
# --ckpt_dir: the path to the checkpoint file.

cd 3d_layout_viewer/
python layout_viewer.py --dataset_dir ../src/custom --dataset custom --mode test --index 0 --layout ../output/ulayout_mp3d_lsun_test/inference_img/panorama_pred_boundary  --vis --ignore_ceiling --ignore_wireframe