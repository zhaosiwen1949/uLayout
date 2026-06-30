#!/usr/bin/env python
'''Map estimated layout boundaries to real SfM 3D points -> a scene point cloud.

Each est_*.json (output/<scene>_uLayout/inference_img/panorama_pred_boundary)
stores a layout boundary as radians per panorama column: boundary[0] is the
ceiling line vc in (-pi/2, 0], boundary[1] the floor line vf in [0, pi/2).

For every boundary point we build a unit bearing in the panorama (get_3d_layout)
frame, rotate it into each rig sub-camera, project with that camera's PINHOLE
intrinsics, and compare to the COLMAP 2D feature points recorded in images.bin.
A match within --pixel-thresh sub-camera pixels yields a point3D_id, which
points3D.bin turns into a colored 3D point. All hits are merged (deduped by
point3D_id) into one .ply.

The panorama<->rig orientation is fixed across frames, so only the per-frame
feature points differ. Uses the same conventions as multi_layout_viewer:
geos(right=+X,up=+Z,front=-Y) -> pano(right=+X,up=+Y,front=+Z) via GEOS_TO_PANO,
pano -> pano_camera0 via Ror=Rx(AX)Ry(AY), camera0 -> cameraX via rigs.bin.
'''
import os
import re
import sys
import json
import glob
import struct
import argparse

import numpy as np

from layout_3d_utils import np_coorx2u
from multi_layout_viewer import GEOS_TO_PANO, reference_rig_rotation


# --- COLMAP binary readers (no pycolmap dependency) --------------------------

# COLMAP camera model_id -> number of params. Only PINHOLE (1) is used here, but
# keep the table so other models are skipped with the right byte count.
CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 5, 7: 4, 8: 5, 9: 6, 10: 12,
}


def read_cameras_bin(path):
    '''Return {cam_id: (W, H, fx, fy, cx, cy)} for PINHOLE cameras.'''
    cams = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack('<iiQQ', f.read(24))
            nparams = CAMERA_MODEL_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack('<%dd' % nparams, f.read(8 * nparams))
            if model_id == 1:                       # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = params
            else:                                   # fall back to fx=fy=params[0]
                fx = fy = params[0]
                cx, cy = w / 2.0, h / 2.0
            cams[cam_id] = (int(w), int(h), fx, fy, cx, cy)
    return cams


def _qvec2rotmat(qvec):
    '''Quaternion (w, x, y, z) -> 3x3 rotation matrix.'''
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_rigs_bin(path):
    '''Return {cam_id: R_sensor_from_rig (3x3)} from a COLMAP rigs.bin.

    The rig's reference sensor (pano_camera0) maps to identity. Each non-ref
    sensor block is 65 bytes: type i32, id u32, has_pose u8, quat 4xf64 (w first),
    trans 3xf64. The header occupies the first 24 bytes (num_rigs u64, then rig
    header + ref sensor type/id); sensor blocks start at byte 24.
    '''
    raw = open(path, 'rb').read()
    ref_type, ref_id = struct.unpack_from('<iI', raw, 16)
    rigs = {ref_id: np.eye(3)}                       # reference sensor = identity
    off = 24
    while off + 9 <= len(raw):
        stype, sid = struct.unpack_from('<iI', raw, off)
        has_pose = raw[off + 8]
        if not has_pose:
            off += 9
            continue
        quat = struct.unpack_from('<4d', raw, off + 9)
        rigs[sid] = _qvec2rotmat(quat)               # sensor_from_rig rotation
        off += 65
    return rigs


def read_points3d_bin(path):
    '''Return (ids_sorted, xyz[N,3] float32, rgb[N,3] uint8) for fast lookup.'''
    ids, xyz, rgb = [], [], []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack('<Q', f.read(8))[0]
            x, y, z = struct.unpack('<3d', f.read(24))
            r, g, b = struct.unpack('<3B', f.read(3))
            f.read(8)                                # reprojection error
            track_len = struct.unpack('<Q', f.read(8))[0]
            f.read(track_len * 8)                    # track (image_id, point2D_idx)
            ids.append(pid)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    ids = np.asarray(ids, dtype=np.uint64)
    order = np.argsort(ids)
    return (ids[order],
            np.asarray(xyz, dtype=np.float32)[order],
            np.asarray(rgb, dtype=np.uint8)[order])


def _name_to_cam_index(name):
    '''pano_cameraN/frame_XXXXX.jpg -> (cam_index N, frame_key frame_XXXXX).'''
    parts = name.split('/')
    sub = parts[0]
    m = re.search(r'pano_camera(\d+)', sub)
    cam_index = int(m.group(1)) if m else 0
    frame_key = os.path.splitext(parts[-1])[0]
    return cam_index, frame_key


def build_feature_cache(images_bin_path, cache_path):
    '''Parse images.bin into per-(frame, cam_id) valid 2D->3D feature records.

    Caches flat arrays to cache_path (npz) so the multi-GB images.bin is parsed
    once. Records only feature points carrying a valid point3D_id. Returns the
    same dict that load_feature_cache yields.
    '''
    print('[transfer] parsing %s (one-off, building cache) ...' % images_bin_path)
    frame_keys = []
    frame_index = {}                                 # frame_key -> idx
    f_idx, f_cam, f_x, f_y, f_p3d = [], [], [], [], []
    with open(images_bin_path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for i in range(n):
            f.read(4)                                # image_id
            f.read(32 + 24)                          # qvec, tvec (unused: rig fixed)
            cam_id = struct.unpack('<i', f.read(4))[0]
            nb = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                nb += c
            name = nb.decode('utf-8')
            npts = struct.unpack('<Q', f.read(8))[0]
            data = f.read(npts * 24)
            _, frame_key = _name_to_cam_index(name)
            if frame_key not in frame_index:
                frame_index[frame_key] = len(frame_keys)
                frame_keys.append(frame_key)
            fi = frame_index[frame_key]
            arr = np.frombuffer(data, dtype=np.dtype(
                [('x', '<f8'), ('y', '<f8'), ('p3d', '<i8')]))
            valid = arr['p3d'] != -1
            k = int(valid.sum())
            if k == 0:
                continue
            f_idx.append(np.full(k, fi, dtype=np.int32))
            f_cam.append(np.full(k, cam_id, dtype=np.int16))
            f_x.append(arr['x'][valid].astype(np.float32))
            f_y.append(arr['y'][valid].astype(np.float32))
            f_p3d.append(arr['p3d'][valid].astype(np.int64))
            if (i + 1) % 5000 == 0:
                print('  ...%d/%d images' % (i + 1, n))
    np.savez(cache_path,
             frame_keys=np.array(frame_keys),
             frame_idx=np.concatenate(f_idx),
             cam_id=np.concatenate(f_cam),
             x=np.concatenate(f_x),
             y=np.concatenate(f_y),
             p3d=np.concatenate(f_p3d))
    print('[transfer] cached %d frames, %d feature records -> %s'
          % (len(frame_keys), len(np.concatenate(f_idx)), cache_path))
    return load_feature_cache(cache_path)


def load_feature_cache(cache_path):
    '''Load the npz cache into {frame_key: {cam_id: ndarray[N,3] (x,y,p3d)}}.'''
    z = np.load(cache_path, allow_pickle=False)
    frame_keys = [str(k) for k in z['frame_keys']]
    fi, cam, x, y, p3d = z['frame_idx'], z['cam_id'], z['x'], z['y'], z['p3d']
    feats = {fk: {} for fk in frame_keys}
    order = np.lexsort((cam, fi))
    fi, cam, x, y, p3d = fi[order], cam[order], x[order], y[order], p3d[order]
    # split into contiguous (frame, cam) groups
    key = fi.astype(np.int64) * 100 + cam.astype(np.int64)
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(key)]])
    for s, e in zip(starts, ends):
        fk = frame_keys[fi[s]]
        feats[fk][int(cam[s])] = np.stack(
            [x[s:e], y[s:e], p3d[s:e].astype(np.float64)], axis=1)
    return feats


# --- boundary -> bearing -> sub-camera pixel matching ------------------------

def boundary_bearings(boundary, curves):
    '''Yield (column, radian v) for the requested boundary curves.

    boundary: [2, W] with row0=ceiling (v<0, up), row1=floor (v>0, down).
    '''
    W = boundary.shape[1]
    rows = []
    if curves in ('ceiling', 'both'):
        rows.append(0)
    if curves in ('floor', 'both'):
        rows.append(1)
    for r in rows:
        for c in range(W):
            yield c, float(boundary[r, c])


def bearing_geos(u, v):
    '''Unit bearing in the get_3d_layout frame for azimuth u, signed pitch v.

    v>0 points down (floor), v<0 up (ceiling); matches get_3d_layout where the
    floor point is [cs*sin u, -cs*cos u, -1.6].
    '''
    cv = np.cos(v)
    return np.array([cv * np.sin(u), -cv * np.cos(u), -np.sin(v)])


def match_boundary_to_point3d(boundary, cams, cam_rot, frame_feats,
                              pano_to_cam0, pixel_thresh, curves):
    '''Return a set of matched point3D_ids for one frame's boundary.

    cam_rot: {cam_index: R_camX_from_cam0}. cams: {cam_id: (W,H,fx,fy,cx,cy)},
    cam_id = cam_index+1. frame_feats: {cam_id: ndarray[N,3] (x,y,p3d)}.
    '''
    thresh_sq = pixel_thresh * pixel_thresh
    hits = set()
    for c, v in boundary_bearings(boundary, curves):
        u = np_coorx2u(c, boundary.shape[1])
        d_cam0 = pano_to_cam0 @ (GEOS_TO_PANO @ bearing_geos(u, v))
        best = None                                  # (dist_sq, p3d)
        for cam_index, R in cam_rot.items():
            cam_id = cam_index + 1
            feats = frame_feats.get(cam_id)
            if feats is None or cam_id not in cams:
                continue
            d = R @ d_cam0
            if d[2] <= 1e-6:
                continue
            W, H, fx, fy, cx, cy = cams[cam_id]
            px = fx * d[0] / d[2] + cx
            py = fy * d[1] / d[2] + cy
            if not (0 <= px < W and 0 <= py < H):
                continue
            dx = feats[:, 0] - px
            dy = feats[:, 1] - py
            dsq = dx * dx + dy * dy
            j = int(np.argmin(dsq))
            if dsq[j] <= thresh_sq and (best is None or dsq[j] < best[0]):
                best = (dsq[j], int(feats[j, 2]))
        if best is not None:
            hits.add(best[1])
    return hits


def write_ply(path, xyz, rgb):
    '''Write a binary little-endian PLY with per-vertex RGB color.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(xyz)
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        'element vertex %d\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n' % n)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        buf = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                 ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        buf['x'], buf['y'], buf['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        buf['r'], buf['g'], buf['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(buf.tobytes())
    print('Saved %d colored points to %s' % (n, path))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    parser.add_argument('--dataset_dir', default='../src/xinghecheng',
                        help='dataset dir containing sparse/0 and img')
    parser.add_argument('--layout',
                        default='../output/xinghecheng_uLayout/inference_img/'
                                'panorama_pred_boundary',
                        help='directory of est_*.json layout boundaries')
    parser.add_argument('--rig-angles', nargs=2, type=float, default=[35.0, -40.0],
                        metavar=('AX', 'AY'),
                        help='Ror = Rx(AX) @ Ry(AY): pano_camera0 -> panorama')
    parser.add_argument('--pixel-thresh', type=float, default=5.0,
                        help='match threshold in sub-camera pixels')
    parser.add_argument('--curves', choices=['floor', 'ceiling', 'both'],
                        default='both', help='which boundary curves to transfer')
    parser.add_argument('--indices', nargs='+', type=int, default=None,
                        help='subset of est indices to process (default: all)')
    parser.add_argument('--out', default=None,
                        help='output PLY path (default: '
                             'output/<scene>_uLayout/inference_ply/boundary_points3d.ply)')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='force rebuild of the images.bin feature cache')
    args = parser.parse_args()

    sparse_dir = os.path.join(args.dataset_dir, 'sparse', '0')
    cameras_bin = os.path.join(sparse_dir, 'cameras.bin')
    images_bin = os.path.join(sparse_dir, 'images.bin')
    points3d_bin = os.path.join(sparse_dir, 'points3D.bin')
    rigs_bin = os.path.join(sparse_dir, 'rigs.bin')
    cache_path = os.path.join(sparse_dir, 'feat2d3d_cache.npz')
    for p in (cameras_bin, images_bin, points3d_bin, rigs_bin):
        if not os.path.isfile(p):
            print('[transfer] missing %s' % p)
            sys.exit(1)

    scene = os.path.basename(os.path.normpath(args.dataset_dir))
    # Default lands at <repo>/output/<scene>_uLayout/inference_ply/... -- the
    # dataset dir is <repo>/src/<scene>, so ../../output resolves to <repo>/output.
    out_path = args.out or os.path.normpath(os.path.join(
        args.dataset_dir, '..', '..',
        'output', '%s_uLayout' % scene, 'inference_ply', 'boundary_points3d.ply'))

    print('[transfer] reading cameras / rigs / points3D ...')
    cams = read_cameras_bin(cameras_bin)
    cam_rot = {cam_id - 1: R for cam_id, R in read_rigs_bin(rigs_bin).items()}
    pids, p_xyz, p_rgb = read_points3d_bin(points3d_bin)
    print('[transfer] %d cameras, %d rig sensors, %d points3D'
          % (len(cams), len(cam_rot), len(pids)))

    if args.rebuild_cache or not os.path.isfile(cache_path):
        feats = build_feature_cache(images_bin, cache_path)
    else:
        print('[transfer] loading feature cache %s' % cache_path)
        feats = load_feature_cache(cache_path)

    Ror = reference_rig_rotation(*args.rig_angles)
    pano_to_cam0 = Ror.T                              # Ror: cam0 -> pano

    est_paths = sorted(
        glob.glob(os.path.join(args.layout, 'est_*.json')),
        key=lambda p: int(re.search(r'est_(\d+)\.json', os.path.basename(p)).group(1)))
    if not est_paths:
        print('[transfer] no est_*.json in %s' % args.layout)
        sys.exit(1)

    matched_ids = set()
    n_frames = 0
    total_cols = 0
    for est_path in est_paths:
        index = int(re.search(r'est_(\d+)\.json', os.path.basename(est_path)).group(1))
        if args.indices is not None and index not in args.indices:
            continue
        est = json.load(open(est_path))
        frame_key = est['img_name']
        frame_feats = feats.get(frame_key)
        if not frame_feats:
            continue
        boundary = np.array(est['boundary'])
        hits = match_boundary_to_point3d(
            boundary, cams, cam_rot, frame_feats, pano_to_cam0,
            args.pixel_thresh, args.curves)
        matched_ids |= hits
        n_frames += 1
        total_cols += boundary.shape[1] * (2 if args.curves == 'both' else 1)
        if n_frames % 500 == 0:
            print('  ...%d frames, %d unique 3D points so far'
                  % (n_frames, len(matched_ids)))

    if not matched_ids:
        print('[transfer] no boundary points matched any feature '
              '(try a larger --pixel-thresh or adjust --rig-angles)')
        sys.exit()

    ids_arr = np.fromiter(matched_ids, dtype=np.uint64, count=len(matched_ids))
    pos = np.searchsorted(pids, ids_arr)
    ok = (pos < len(pids)) & (pids[np.clip(pos, 0, len(pids) - 1)] == ids_arr)
    pos = pos[ok]
    xyz, rgb = p_xyz[pos], p_rgb[pos]
    print('[transfer] %d frames, %d boundary columns scanned, %d unique 3D points'
          % (n_frames, total_cols, len(xyz)))
    write_ply(out_path, xyz, rgb)


if __name__ == '__main__':
    main()
