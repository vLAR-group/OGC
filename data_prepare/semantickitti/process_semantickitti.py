import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import tqdm
import argparse
import numpy as np

from data_prepare.semantickitti import semantickitti_util as utils
from utils.data_util import fps_downsample


parser = argparse.ArgumentParser()
parser.add_argument('data_root', type=str, help='Root path for the dataset')
parser.add_argument('--visualize', dest='visualize', default=False, action='store_true', help='Visualization for check')
args = parser.parse_args()

SRC_DIR = osp.join(args.data_root, 'sequences')
sequence_ids = list(range(11))

SAVE_DIR = osp.join(args.data_root, 'downsampled')
os.makedirs(SAVE_DIR, exist_ok=True)

img_width, img_height = 1242, 375
clip_distance = 2.0
depth_thresh = 35.0
n_sample_point = 8192

selected_class_ids = [10, 18, 252, 258]     # ['car', 'truck', 'moving-car', 'moving-truck']

if args.visualize:
    import open3d as o3d
    from utils.visual_util import build_pointcloud, build_bbox3d


for seq_id in sequence_ids:
    seq_dir = osp.join(SRC_DIR, '%02d'%(seq_id))
    lidar_dir = osp.join(seq_dir, "velodyne")
    label_dir = osp.join(seq_dir, "labels")
    calib_file = osp.join(seq_dir, "calib.txt")
    calib = utils.Calibration(calib_file)

    seq_sample_ids = list(range(len(os.listdir(label_dir))))
    print('Processing sequence %02d'%(seq_id))
    pbar = tqdm.tqdm(total=len(seq_sample_ids))
    for sid in seq_sample_ids:
        # Load point cloud
        pc_file = osp.join(lidar_dir, '%06d.bin'%(sid))
        pc_velo = np.fromfile(pc_file, dtype=np.float32).reshape((-1, 4))[:, :3]

        # Load segmentation label
        label_file = osp.join(label_dir, '%06d.label'%(sid))
        label = np.fromfile(label_file, dtype=np.int32).reshape(-1)
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label

        # Extract front-view point cloud from velodyne data
        pts_3d_ref = calib.project_velo_to_ref(pc_velo)
        pts_2d_depth = calib.project_ref_to_image(pts_3d_ref)
        pts_2d, depth = pts_2d_depth[:, :2], pts_2d_depth[:, 2]
        fov_inds = (
                (pts_2d[:, 0] < img_width)
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img_height)
                & (pts_2d[:, 1] >= 0)
        )
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
        pc = pts_3d_ref[fov_inds]
        sem_label, inst_label =sem_label[fov_inds], inst_label[fov_inds]
        pc[:, :2] *= -1.
        pc = pc.astype(np.float32)
        # Filter by depth
        not_far = (pc[:, 2] < depth_thresh)
        pc = pc[not_far]
        sem_label, inst_label =sem_label[not_far], inst_label[not_far]

        # FPS downsample
        fps_idx = fps_downsample(pc, n_sample_point=n_sample_point)
        pc = pc[fps_idx]
        sem_label, inst_label = sem_label[fps_idx], inst_label[fps_idx]

        # Only keep instances of selected classes
        segm = np.zeros_like(inst_label)
        keep = np.in1d(sem_label, selected_class_ids)
        segm[keep] = inst_label[keep]
        _, segm = np.unique(segm, return_inverse=True)

        # Visualize if needed
        if args.visualize:
            pcds = []
            pcds.append(build_pointcloud(pc, inst_label, with_background=True).translate([-50, 0, 0]))
            pcds.append(build_pointcloud(pc, segm, with_background=True))
            o3d.visualization.draw_geometries(pcds)

        # Save
        save_path = osp.join(SAVE_DIR, '%02d_%06d'%(seq_id, sid))
        os.makedirs(save_path, exist_ok=True)
        np.save(osp.join(save_path, 'pc.npy'), pc)
        np.save(osp.join(save_path, 'segm.npy'), segm)

        pbar.update(1)