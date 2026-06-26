import os
import json
import struct
import numpy as np

from PIL import Image

import torch
import torch.utils.data as data


class CustomPanoDataset(data.Dataset):
    def __init__(self, root_dir, mode='train',
                flip=False, rotate=False, gamma=False, stretch=False, crop=False, req_depth=False, resize=False,
                p_base=0.96, max_stretch=2.0,
                normcor=False, return_cor=False, return_path=False):
    
        assert mode in ('train', 'test'), 'mode should be train/val/test'
        if mode == 'train':
            flip = True
            rotate = True
            gamma = True
            stretch = True

        # mode no longer adds a sub-directory level; images live directly under
        # <root_dir>/img. No label_cor (ground-truth) is read or provided.
        self.img_dir = os.path.join(root_dir, 'img')
        self.img_fnames = sorted([
            fname for fname in os.listdir(self.img_dir)
            if fname.endswith('.jpg') or fname.endswith('.png')
        ])

        self.mode = mode
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.resize = resize
        self.crop = crop
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.return_cor = return_cor
        self.return_path = return_path
        self.req_depth = req_depth

        # Camera extrinsics from COLMAP SfM (optional, used by multi_layout_viewer)
        self.images_bin_path = os.path.join(root_dir, 'images.bin')
        self.camera_poses_path = os.path.join(root_dir, 'camera_poses.json')
        self.camera_poses = self._load_or_build_camera_poses()

    def _load_or_build_camera_poses(self):
        '''Load cached camera poses, or build them from COLMAP images.bin.

        Returns dict {frame_key: {"qvec", "tvec", "R", "camera_center"}} holding
        the raw COLMAP extrinsics of pano_camera0 (world->camera, SfM units, no
        angle processing), or {} when no SfM data exists.
        '''
        if os.path.isfile(self.camera_poses_path):
            with open(self.camera_poses_path) as f:
                data = json.load(f)
            return data.get('poses', data)
        if not os.path.isfile(self.images_bin_path):
            print('[CustomPanoDataset] %s not found, skip camera-pose generation'
                  % self.images_bin_path)
            return {}
        print('[CustomPanoDataset] parsing %s ...' % self.images_bin_path)
        poses, meta = self._build_camera_poses_from_bin(self.images_bin_path)
        with open(self.camera_poses_path, 'w') as f:
            json.dump({'metadata': meta, 'poses': poses}, f, indent=2)
        print('[CustomPanoDataset] saved %d camera poses to %s'
              % (len(poses), self.camera_poses_path))
        return poses

    def get_camera_pose(self, img_name):
        '''Return raw pano_camera0 extrinsics for a frame, or None.

        {"qvec": [w,x,y,z], "tvec": [..], "R": 3x3 (world->camera),
         "camera_center": [..] (-R.T @ tvec, SfM world units)}.
        '''
        return self.camera_poses.get(img_name)

    @staticmethod
    def _qvec2rotmat(qvec):
        '''Quaternion (w, x, y, z) to 3x3 rotation matrix (world -> camera).'''
        w, x, y, z = qvec
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ])

    @staticmethod
    def _read_images_bin(path):
        '''Parse a COLMAP images.bin and yield (name, qvec, tvec, R, cam_center).

        Binary layout (COLMAP standard):
          num_reg_images: uint64
          per image: image_id int32, qvec 4xfloat64 (w,x,y,z), tvec 3xfloat64,
                     camera_id int32, name '\\0'-terminated,
                     num_points2D uint64, points2D num_points2D x 24 bytes (skipped)
        '''
        with open(path, 'rb') as f:
            num_images = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_images):
                f.read(4)  # image_id
                qvec = struct.unpack('<4d', f.read(32))
                tvec = struct.unpack('<3d', f.read(24))
                f.read(4)  # camera_id
                name_bytes = b''
                while True:
                    c = f.read(1)
                    if c == b'\x00':
                        break
                    name_bytes += c
                name = name_bytes.decode('utf-8')
                num_points2d = struct.unpack('<Q', f.read(8))[0]
                f.seek(num_points2d * 24, 1)

                R = CustomPanoDataset._qvec2rotmat(qvec)  # world -> camera
                cam_center = -R.T @ np.array(tvec)        # camera position in world
                yield name, np.array(qvec), np.array(tvec), R, cam_center

    @staticmethod
    def _build_camera_poses_from_bin(path):
        '''Collect the raw COLMAP extrinsics of pano_camera0 per pano frame.

        No angle processing: each frame stores pano_camera0's quaternion
        (w,x,y,z), translation, world->camera rotation R and camera center
        (-R.T @ tvec), all in SfM world units. pano_camera0 is one perspective
        crop of a fixed 6-view rig (the inter-camera relative rotations are
        identical across all frames), so it defines the panorama pose up to a
        fixed pano<->camera rotation that the viewer applies.

        Returns (poses, metadata).
        '''
        poses = {}
        for name, qvec, tvec, R, center in CustomPanoDataset._read_images_bin(path):
            parts = name.split('/')
            sub_cam = parts[0] if len(parts) > 1 else ''
            if sub_cam != 'pano_camera0':
                continue
            frame_key = os.path.splitext(parts[-1])[0]
            poses[frame_key] = {
                'qvec': [float(v) for v in qvec],   # (w, x, y, z), world->camera
                'tvec': [float(v) for v in tvec],
                'R': R.tolist(),                    # world->camera 3x3
                'camera_center': center.tolist(),   # -R.T @ tvec, SfM world units
            }

        metadata = {
            'reference_camera': 'pano_camera0',
            'num_frames': len(poses),
            'note': 'raw COLMAP extrinsics of pano_camera0 (world->camera), '
                    'no angle processing',
            'units': 'SfM (apply --scale in multi_layout_viewer to convert to meters)',
        }
        return poses, metadata

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        # Read image only. No label_cor ground truth is read or returned.
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
        img_name = self.img_fnames[idx].split('.')[0]
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.

        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())

        output = {
            'img_name': img_name,
            'img': x,
        }

        return output
        