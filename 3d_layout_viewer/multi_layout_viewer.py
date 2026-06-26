import os
import re
import json
import numpy as np
import open3d as o3d
import glob
import sys
from layout_3d_utils import get_3d_layout, np_coorx2u
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.custom_pano_dataset import CustomPanoDataset


# --- Rig rotation between pano_camera0 and the panorama coordinate frame ------
#
# pano_camera0 is mounted at a fixed orientation relative to the panorama, so a
# rig rotation Ror = Rx(AX) @ Ry(AY) (degrees) is needed to convert its COLMAP
# extrinsics into the panorama frame, following rig_reconstruction.jpeg.


# Change of basis from the get_3d_layout frame to the reference's pano coordinate
# frame. get_3d_layout uses right=+X, up=+Z, panorama-center/front (u=0) = -Y;
# the reference's "pano coordinate" (the frame Ror maps the camera into) is taken
# to be Y-up: right=+X, up=+Y, front=+Z. This maps geos +Z(up)->+Y, -Y(front)->+Z,
# +X(right)->+X (a proper rotation, det +1). Adjust here if the reference pano
# convention differs (confirm by rendering).
GEOS_TO_PANO = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
])


def reference_rig_rotation(ax_deg, ay_deg):
    '''Ror = Rx(ax) @ Ry(ay): rig-ref (pano_camera0) camera -> pano coordinate.

    Matches rig_reconstruction.jpeg:
        Quaternion(axis=(1,0,0), angle=ax) * Quaternion(axis=(0,1,0), angle=ay)
    '''
    ax = ax_deg * np.pi / 180.0
    ay = ay_deg * np.pi / 180.0
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(ax), -np.sin(ax)],
                   [0.0, np.sin(ax),  np.cos(ax)]])
    Ry = np.array([[ np.cos(ay), 0.0, np.sin(ay)],
                   [ 0.0,        1.0, 0.0],
                   [-np.sin(ay), 0.0, np.cos(ay)]])
    return Rx @ Ry


def build_world_transform(pose, scale, Ror):
    '''4x4 placing a panorama-local room mesh into the world frame.

    The mesh is in the get_3d_layout frame (origin at the panorama optical
    center, floor at z=-1.6), so it is first rotated into the reference's pano
    coordinate frame by GEOS_TO_PANO. pano_camera0's COLMAP extrinsics are then
    applied following rig_reconstruction.jpeg:

        M     = [[GEOS_TO_PANO, 0], [0, 1]]        # get_3d_layout -> pano
        Twc   = [[R.T, -R.T@tvec], [0, 1]]         # camera -> world (R is world->camera)
        Tor   = [[Ror, 0], [0, 1]]                 # rig-ref camera -> pano
        T     = Tor @ Twc @ Tor_inv @ M            # get_3d_layout -> world

    The full per-frame camera rotation is applied; only the translation is
    scaled (layout is metric, SfM translations are in arbitrary units).
    '''
    R = np.asarray(pose['R'])             # COLMAP world -> camera
    tvec = np.asarray(pose['tvec'])
    # COLMAP's R is world->camera, so the camera->world transform Twc uses R's
    # inverse (R.T) and the corrected translation -- the camera center
    # C = -R.T @ tvec (not tvec itself).
    Twc = np.eye(4)
    Twc[:3, :3] = R.T
    Twc[:3, 3] = -R.T @ tvec
    Tor = np.eye(4)
    Tor[:3, :3] = Ror
    Tor_inv = np.linalg.inv(Tor)
    M = np.eye(4)
    M[:3, :3] = GEOS_TO_PANO              # get_3d_layout frame -> pano coordinate
    # Convert the room from the get_3d_layout frame into the pano frame (M), then
    # conjugate the camera->world transform to land in world:
    T = Tor @ Twc @ Tor_inv @ M          # get_3d_layout -> world
    T[:3, 3] *= scale
    return T


def floor_outline_3d(est_layout):
    '''Wall-base outline in the panorama-local frame (floor plane z=-1.6).

    Per column the floor boundary angle vf (>0, below horizon) gives the
    horizontal wall distance cs = 1.6 / tan(vf) (same convention as
    layout_2_depth), placed at azimuth u with x = cs*sin(u), y = -cs*cos(u),
    z = -1.6. Returns [W, 3] local points (one per panorama column).
    '''
    W = est_layout.shape[1]
    us = np_coorx2u(np.arange(W), W)
    cs = 1.6 / np.tan(est_layout[1])
    return np.stack([cs * np.sin(us), -cs * np.cos(us), np.full(W, -1.6)], 1)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_dir',
                        required=True,
                        help='dataset dir')
    parser.add_argument('--dataset',
                        default='custom',
                        choices=['custom'],
                        help='only custom dataset provides SfM camera poses')
    parser.add_argument('--mode',
                        default='test',
                        help='train or test')
    parser.add_argument('--layout',
                        required=True,
                        help='Directory containing est_*.json layout estimations')
    parser.add_argument('--indices',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Subset of est indices to render (default: all)')
    parser.add_argument('--scale',
                        type=float,
                        default=1.0,
                        help='Multiply SfM translations by this factor to convert to meters')
    parser.add_argument('--rig-angles',
                        nargs=2,
                        type=float,
                        default=[35.0, -40.0],
                        metavar=('AX', 'AY'),
                        help='rig_reconstruction.jpeg rig rotation Ror = Rx(AX) @ Ry(AY) '
                             '(degrees) mapping pano_camera0 -> panorama frame')
    parser.add_argument('--plan2d', action='store_true',
                        help='Show a top-down 2D floor plan (project onto the scene floor plane)')
    parser.add_argument('--out',
                        help='Export combined mesh as PLY (or image, e.g. plan.png, with --plan2d)')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',
                        help='Skip rendering ceiling')
    parser.add_argument('--ignore_wall', action='store_true',
                        help='Skip rendering wall')
    parser.add_argument('--ignore_wireframe', action='store_true',
                        help='Skip rendering wireframe')
    args = parser.parse_args()

    if not args.out and not args.vis:
        print('You may want to export (via --out) or visualize (via --vis)')
        sys.exit()

    # Builds/loads <dataset_dir>/camera_poses.json from images.bin
    dataset = CustomPanoDataset(args.dataset_dir, mode=args.mode)

    est_paths = sorted(
        glob.glob(os.path.join(args.layout, 'est_*.json')),
        key=lambda p: int(re.search(r'est_(\d+)\.json', os.path.basename(p)).group(1)))
    if not est_paths:
        print('No est_*.json found in %s' % args.layout)
        sys.exit()

    # Rig rotation from rig_reconstruction.jpeg (degrees).
    Ror = reference_rig_rotation(*args.rig_angles)
    print('rig: Ror = Rx(%g) @ Ry(%g) deg, get_3d_layout->pano = GEOS_TO_PANO'
          % tuple(args.rig_angles))

    geometries = []
    meshes = []
    rooms_2d = []  # (index, img_name, outline [W, 2], camera xy) in world frame
    for est_path in est_paths:
        index = int(re.search(r'est_(\d+)\.json', os.path.basename(est_path)).group(1))
        if args.indices is not None and index not in args.indices:
            continue
        with open(est_path, 'r') as f:
            est = json.load(f)

        img_name = est['img_name']
        dataset_name = dataset.img_fnames[index].split('.')[0]
        if img_name != dataset_name:
            print('[warn] est_%d img_name %s != dataset[%d] %s'
                  % (index, img_name, index, dataset_name))

        cor_id = np.array(est['corner'])
        est_layout = np.array(est['boundary'])

        pose = dataset.get_camera_pose(img_name)
        if pose is None:
            print('[warn] no camera pose for %s, placing at origin' % img_name)
            T = np.eye(4)
        else:
            T = build_world_transform(pose, args.scale, Ror)
            c = T[:3, 3]
            print('est_%d %s: camera_center=(%.2f, %.2f, %.2f)'
                  % (index, img_name, c[0], c[1], c[2]))

        if args.plan2d:
            # Transform the local floor outline into world. The scene is globally
            # rotated (its floor is not the world X-Z plane), so keep the 3D world
            # points and the room's up direction here; the actual floor-plane
            # projection happens after the loop once the common up is known.
            local = floor_outline_3d(est_layout)              # [W, 3]
            world = local @ T[:3, :3].T + T[:3, 3]            # [W, 3]
            up = T[:3, :3] @ np.array([0.0, 0.0, 1.0])        # room up (+Z) in world
            rooms_2d.append((index, img_name, world, T[:3, 3].copy(), up))
            continue

        equirect_texture = dataset[index]['img'].numpy().transpose(1, 2, 0) * 255
        geos = get_3d_layout(equirect_texture, est_layout, cor_id,
                             ignore_floor=args.ignore_floor,
                             ignore_ceiling=args.ignore_ceiling,
                             ignore_wall=args.ignore_wall,
                             ignore_wireframe=args.ignore_wireframe)
        for g in geos:
            g.transform(T)
            geometries.append(g)
            if isinstance(g, o3d.geometry.TriangleMesh):
                meshes.append(g)

    if args.plan2d:
        # The scene has an overall rotation, so its floor is not the world X-Z
        # plane. Project onto the actual floor plane: take the common up
        # direction (average of the rooms' up vectors) and an orthonormal
        # in-plane basis (e1, e2) perpendicular to it. (u . e1 = u . e2 = 0, so
        # dropping the up component gives a true top-down floor plan.)
        up = np.mean([u for _, _, _, _, u in rooms_2d], axis=0)
        up = up / np.linalg.norm(up)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(seed @ up) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        e1 = seed - (seed @ up) * up
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        basis = np.column_stack([e1, e2])                     # [3, 2]
        rooms_2d = [(index, img_name, world @ basis, cam @ basis)
                    for index, img_name, world, cam, _ in rooms_2d]

        # tab10-like palette
        palette = [
            [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17],
            [0.84, 0.15, 0.16], [0.58, 0.40, 0.74], [0.55, 0.34, 0.29],
            [0.89, 0.47, 0.76], [0.50, 0.50, 0.50], [0.74, 0.74, 0.13],
            [0.09, 0.75, 0.81],
        ]
        plan_geometries = []
        for i, (index, img_name, outline, cam_xy) in enumerate(rooms_2d):
            color = palette[i % len(palette)]
            N = len(outline)
            # Filled room polygon: the outline is star-shaped around the
            # camera (one wall distance per azimuth), so a triangle fan from
            # the camera center triangulates it exactly. Each room gets its
            # own depth so overlapping fills don't z-fight (earlier est on top).
            fill_z = -0.01 * (i + 1)
            verts = np.zeros((N + 1, 3))
            verts[0, :2] = cam_xy
            verts[1:, :2] = outline
            verts[:, 2] = fill_z
            tris = [[0, 1 + j, 1 + (j + 1) % N] for j in range(N)]
            fill = o3d.geometry.TriangleMesh()
            fill.vertices = o3d.utility.Vector3dVector(verts)
            fill.triangles = o3d.utility.Vector3iVector(tris)
            fill.paint_uniform_color(color)
            plan_geometries.append(fill)
            # Wall outline slightly above the fills for crisp borders
            points = np.concatenate([outline, np.full((N, 1), 0.01)], 1)
            lines = [[j, (j + 1) % N] for j in range(N)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(
                [[c * 0.5 for c in color]] * len(lines))
            plan_geometries.append(line_set)
            # Camera position marker
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            marker.translate([cam_xy[0], cam_xy[1], 0.05])
            marker.paint_uniform_color([c * 0.5 for c in color])
            plan_geometries.append(marker)

        if args.out:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 10))
            for i, (index, img_name, outline, cam_xy) in enumerate(rooms_2d):
                color = palette[i % len(palette)]
                closed = np.vstack([outline, outline[:1]])
                ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.4,
                        label='est_%d %s' % (index, img_name))
                ax.plot(closed[:, 0], closed[:, 1], '-', color=color, lw=1.5)
                ax.plot(cam_xy[0], cam_xy[1], 'o', color=color, ms=6)
                ax.annotate(str(index), cam_xy, textcoords='offset points',
                            xytext=(5, 5), color=color)
            ax.set_aspect('equal')
            ax.grid(True, ls='--', alpha=0.5)
            ax.set_xlabel('floor e1 (m)')
            ax.set_ylabel('floor e2 (m)')
            ax.set_title('Top-down floor plan (floor-plane projection, scale=%g)' % args.scale)
            ax.legend(loc='best', fontsize=8)
            fig.savefig(args.out, dpi=200, bbox_inches='tight')
            print('Saved 2D floor plan to %s' % args.out)

        if args.vis:
            # Interactive open3d viewer looking straight down at the xy plane
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='2D floor plan')
            for g in plan_geometries:
                vis.add_geometry(g)
            vis.get_render_option().mesh_show_back_face = True
            # Top-down camera centered on the rendered rooms (with --indices
            # the geometry can sit far away from the origin), at a distance
            # derived from the camera intrinsics so everything fits on screen.
            all_xy = np.vstack([outline for _, _, outline, _ in rooms_2d])
            cx, cy = (all_xy.min(0) + all_xy.max(0)) / 2
            ext_x, ext_y = all_xy.max(0) - all_xy.min(0)
            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            K = params.intrinsic.intrinsic_matrix
            w, h = params.intrinsic.width, params.intrinsic.height
            dist = 1.2 * max(ext_x * K[0][0] / w, ext_y * K[1][1] / h)
            # Camera above (cx, cy) looking down -z, world +y up on screen
            params.extrinsic = np.array([
                [1,  0,  0, -cx],
                [0, -1,  0,  cy],
                [0,  0, -1, dist],
                [0,  0,  0,  1.0],
            ])
            ctr.convert_from_pinhole_camera_parameters(params)

            def lock_topdown(v):
                # Re-pin the view direction every frame: out-of-plane rotation
                # snaps back to top-down, while the up vector (in-plane
                # rotation), pan and zoom from mouse drags are kept.
                v.get_view_control().set_front([0, 0, 1])
                return False

            vis.register_animation_callback(lock_topdown)
            vis.run()
            vis.destroy_window()
        sys.exit()

    if args.out:
        combined = o3d.geometry.TriangleMesh()
        for mesh in meshes:
            combined += mesh
        o3d.io.write_triangle_mesh(args.out, combined)
        print('Saved combined mesh to %s' % args.out)

    if args.vis:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        o3d.visualization.draw_geometries(geometries, mesh_show_back_face=True)
