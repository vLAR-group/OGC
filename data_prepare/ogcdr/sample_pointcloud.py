import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import trimesh

from utils.data_util import fps_downsample


parser = argparse.ArgumentParser()
parser.add_argument('data_root', type=str, help='Root path for the dataset')
parser.add_argument('--n_sample_point_fps', type=int, default=2048, help='Number of points for point cloud sampling')
parser.add_argument('--keep_background', dest='keep_background', default=False, action='store_true', help='Keep the background in sampled point clouds or not')
args = parser.parse_args()

ground_thickness = 0.01
ground_height = -0.5
ground_level = ground_height + ground_thickness
wall_thickness = 0.01

# Hyperparams for sampling point cloud
n_sample_point = 100000
n_sample_point_fps = args.n_sample_point_fps
keep_background = args.keep_background

# Data path
data_root = osp.join(args.data_root, 'mesh')
pc_root = osp.join(args.data_root, 'data')
data_ids = sorted(os.listdir(data_root))
n_frame = 4
save_root = '/home/ziyang/Desktop/Datasets/OGC_DynamicRoom_p%d/data'%(n_sample_point_fps)
os.makedirs(save_root, exist_ok=True)


def sample_pointcloud(meshes, walls, ground, xz_range):
    """
    Sample point cloud from mesh list.
    """
    n_object = len(meshes)
    if keep_background:
        meshes = meshes + [ground] + walls

    c_vol = np.array([mesh.area for mesh in meshes])
    c_vol /= sum(c_vol)
    n_points = [int(c * n_sample_point) for c in c_vol]

    # Sample points from mesh surfaces
    points, segms = [], []
    for i, mesh in enumerate(meshes):
        pi, _ = trimesh.sample.sample_surface_even(mesh, n_points[i])
        if i < n_object:
            # Foreground object has segment ids starting from 1
            segm = (i + 1) * np.ones(pi.shape[0], dtype=np.int16)
        else:
            segm = np.zeros(pi.shape[0], dtype=np.int16)
        points.append(pi)
        segms.append(segm)
    points = np.concatenate(points, axis=0).astype(np.float32)
    segms = np.concatenate(segms, axis=0).astype(np.int16)

    # Remove the thickness of ground & wall from pointcloud
    mask = points[:, 1] > (ground_level - 1e-4)
    mask &= points[:, 2] > (- xz_range[1] / 2. + wall_thickness - 1e-4)
    mask &= points[:, 0] > (- xz_range[0] / 2. + wall_thickness - 1e-4)
    mask &= points[:, 2] < (+ xz_range[1] / 2. - wall_thickness + 1e-4)
    mask &= points[:, 0] < (+ xz_range[0] / 2. - wall_thickness + 1e-4)
    points = points[mask]
    segms = segms[mask]

    # FPS downsample
    fps_idx = fps_downsample(points, n_sample_point=n_sample_point_fps)
    points = points[fps_idx]
    segms = segms[fps_idx]
    return points, segms


# Iterate over the dataset
pbar = tqdm(total=len(data_ids))
for data_id in data_ids:
    n_object = int(data_id[:2])
    data_path = osp.join(data_root, data_id)
    save_path = osp.join(save_root, data_id)
    os.makedirs(save_path, exist_ok=True)

    # Load the meta info
    meta_file = osp.join(data_path, 'meta.pkl')
    with open(meta_file, 'rb') as f:
        item_dict = pickle.load(f)

    for frame_id in range(n_frame):
        # Load the objects
        meshes = []
        for obj_id in range(n_object):
            model_path = osp.join(data_path, 'object_%02d_%02d.obj'%(frame_id, obj_id))
            mesh = trimesh.load(model_path, force='mesh')
            meshes.append(mesh)

        # Load ground and walls
        ground_path = osp.join(data_path, 'ground.obj')
        ground = trimesh.load(ground_path, force='mesh')
        walls = []
        for wall_id in range(4):
            wall_path = osp.join(data_path, 'wall_%02d.obj'%(wall_id))
            wall = trimesh.load(wall_path, force='mesh')
            walls.append(wall)

        # Sample the point cloud
        points, segms = sample_pointcloud(meshes, walls, ground, xz_range=item_dict['xz_groundplane_range'])
        pose = np.load(osp.join(pc_root, data_id, 'pose_%02d.npy'%(frame_id)))

        # import open3d as o3d
        # from utils.visual_util import build_pointcloud
        # pcds = [build_pointcloud(points, segms)]
        # o3d.visualization.draw_geometries(pcds)

        # Save
        np.save(osp.join(save_path, 'pc_%02d.npy'%(frame_id)), points)
        np.save(osp.join(save_path, 'segm_%02d.npy'%(frame_id)), segms)
        np.save(osp.join(save_path, 'pose_%02d.npy'%(frame_id)), pose)

    pbar.update(1)

pbar.close()