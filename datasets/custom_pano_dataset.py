import pdb
import os
import json
import struct
import numpy as np
import sys
import cv2

from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
from utils.pp_utils import boundary2depth_ceilingfloor
from datasets import panostretch

import torch
import torch.utils.data as data
from datasets.mp3d_hn_ori_dataset import PanoCorBonDataset


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

        root_dir = os.path.join(root_dir, mode)
        self.img_dir = os.path.join(root_dir, 'img')
        self.cor_dir = os.path.join(root_dir, 'label_cor')
        self.img_fnames = sorted([
            fname for fname in os.listdir(self.img_dir)
            if fname.endswith('.jpg') or fname.endswith('.png')
        ])
        self.txt_fnames = sorted([
            fname for fname in os.listdir(self.cor_dir)
            if fname.endswith('.txt')
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

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)),\
                '%s not found' % os.path.join(self.cor_dir, fname)

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
        # Read image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
        img_name = self.img_fnames[idx].split('.')[0]
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
        H, W = img.shape[:2]

        # Read ground truth corners
        with open(os.path.join(self.cor_dir,
                               self.txt_fnames[idx])) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

            # Corner with minimum x should at the beginning
            cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

            # Detect occlusion
            occlusion = find_occlusion(cor[::2].copy()).repeat(2)
            assert (np.abs(cor[0::2, 0] - cor[1::2, 0]) > W/100).sum() == 0, img_path
            assert (cor[0::2, 1] > cor[1::2, 1]).sum() == 0, img_path

        # Stretch augmentation
        if self.stretch:
            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            img, cor = panostretch.pano_stretch(img, cor, kx, ky)

        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = cor_2_1d(cor, H, W)

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            bon = np.flip(bon, axis=1)
            cor[:, 0] = img.shape[1] - 1 - cor[:, 0]

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            bon = np.roll(bon, dx, axis=1)
            cor[:, 0] = (cor[:, 0] + dx) % img.shape[1]

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p

        # Prepare 1d wall-wall probability
        corx = cor[~occlusion, 0]
        dist_o = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1),
                       'minkowski',
                       p=1)
        dist_r = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) + img.shape[1],
                       'minkowski',
                       p=1)
        dist_l = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) - img.shape[1],
                       'minkowski',
                       p=1)
        dist = np.min([dist_o, dist_r, dist_l], 0)
        nearest_dist = dist.min(0)
        y_cor = (self.p_base ** nearest_dist).reshape(1, -1)
        gt_type = 0
        u_range = np.ones_like(bon[0])
        eval_range = np.ones_like(bon)

        # get depth
        depth = boundary2depth_ceilingfloor(bon, gt_type, eval_range)
        depth = torch.FloatTensor(depth.copy())

        # Convert all data to tensor
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        bon = torch.FloatTensor(bon.copy())
        u_range = torch.FloatTensor(u_range.copy())
        eval_range = torch.FloatTensor(eval_range.copy())
        y_cor = torch.FloatTensor(y_cor.copy())
        # gt_type : 0 for both, 1 for ceiling, 2 for floor
        gt_type = torch.FloatTensor([gt_type])
        v_shift_pixel = torch.FloatTensor([0])
        new_cor = np.zeros((50, 2), dtype=np.float32)
        new_cor[:len(cor)] = cor
        new_cor = torch.FloatTensor(new_cor)

        output = {
            'img_name': img_name,
            'img': x,
            'corner': new_cor.T,
            'boundary': bon,
            'depth': depth,
            'u_range': u_range,
            'eval_range': eval_range,
            'gt_type': gt_type,
            'v_shift': v_shift_pixel,
        }

        return output


def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])

    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def find_occlusion(coor):
    u = panostretch.coorx2u(coor[:, 0])
    v = panostretch.coory2v(coor[:, 1])
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)


def visualize_a_data(x, y_bon, y_cor):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_bon = y_bon.numpy()
    y_bon = ((y_bon / np.pi + 0.5) * x.shape[0]).round().astype(int)
    y_cor = y_cor.numpy()

    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] = y_cor[0][None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255

    img_bon = (x.copy() * 0.5).astype(np.uint8)
    y1 = np.round(y_bon[0]).astype(int)
    y2 = np.round(y_bon[1]).astype(int)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255

    return np.concatenate([gt_cor, img_pad, img_bon], 0)
        