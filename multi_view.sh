#!/usr/bin/env bash
# Show all estimated layouts in one shared 3D space, placed by pano_camera0's
# COLMAP extrinsics via the rig rotation Ror = Rx(AX) @ Ry(AY) (default 35/-40)
# plus the GEOS_TO_PANO change of basis. NOTE: each sequence (frame_00,
# frame_01, ...) is an independent COLMAP world -- render one sequence at a time.
# --scale:      multiply SfM translations to convert to meters (tune until room spacing looks right)
# --indices:    optional subset, e.g. --indices 0 1
# --rig-angles: rig rotation angles AX AY in degrees (default 35 -40)
# --plan2d:     show a top-down 2D floor plan (project onto the scene floor plane)
# --out:        optional export: combined PLY, or an image (e.g. plan.png) with --plan2d

cd 3d_layout_viewer/
python multi_layout_viewer.py \
    --dataset_dir ../src/custom \
    --dataset custom \
    --mode test \
    --layout ../output/ulayout_mp3d_lsun_test/inference_img/panorama_pred_boundary \
    --scale 1.0 --vis --ignore_ceiling --ignore_wireframe \
    --rig-angles 35 -40 --plan2d

