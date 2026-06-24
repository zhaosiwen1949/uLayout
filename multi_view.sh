#!/usr/bin/env bash
# Show all estimated layouts in one shared 3D space, placed by pano_camera0's
# COLMAP extrinsics. By default the rig rotation is gravity-calibrated per
# capture sequence, so rooms stand upright (walls vertical) and tile into a
# coherent floor plan. NOTE: each sequence (frame_00, frame_01, ...) is an
# independent COLMAP world -- render one sequence's layouts at a time.
# --scale:      multiply SfM translations to convert to meters (tune until room spacing looks right)
# --indices:    optional subset, e.g. --indices 0 1
# --rig-angles: use the literal rig_reconstruction.jpeg rig instead, e.g. --rig-angles 35 -40
# --plan2d:     show a top-down 2D floor plan (project world onto the X-Z plane)
# --out:        optional export: combined PLY, or an image (e.g. plan.png) with --plan2d

cd 3d_layout_viewer/
python multi_layout_viewer.py
    --dataset_dir ../src/custom \
    --dataset custom \
    --mode test \
    --layout ../output/ulayout_mp3d_lsun_test/inference_img/panorama_pred_boundary \
    --scale 1.0 --vis --ignore_ceiling --ignore_wireframe \
    --rig-angles 35 -40 --plan2d

