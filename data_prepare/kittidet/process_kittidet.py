import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import tqdm
import argparse
import numpy as np

from data_prepare.kittidet import kittidet_util as utils
from utils.data_util import fps_downsample


def get_lidar(idx, dtype=np.float32, n_vec=4):
    lidar_filename = os.path.join(lidar_root, "%06d.bin" % (idx))
    return utils.load_velo_scan(lidar_filename, dtype, n_vec)

def get_image(idx):
    img_filename = os.path.join(image_root, "%06d.png" % (idx))
    return utils.load_image(img_filename)

def get_calibration(idx):
    calib_filename = os.path.join(calib_root, "%06d.txt" % (idx))
    return utils.Calibration(calib_filename)

def get_label_objects(idx):
    label_filename = os.path.join(label_root, "%06d.txt" % (idx))
    return utils.read_label(label_filename)


def box_to_segm(points, objects, relax=0.01):
    """
    :param points: (N, 3).
    :param objects: list of Object3d.
    :return:
        segm: (N,).
    """
    n_point = points.shape[0]
    segm = np.zeros(n_point, dtype=np.int32)

    pc = np.copy(points)
    pc[:, :2] *= -1.

    for sid, obj in enumerate(objects):
        if obj.type != 'Car':
            continue

        R = utils.roty(-obj.ry)
        transl = obj.t
        l, w, h = obj.l, obj.w, obj.h

        pc_tr = pc - transl
        pc_tr = np.einsum('ij,nj->ni', R, pc_tr)

        # Select points within bounding box
        within_box_x = np.logical_and(pc_tr[:, 0] > (-l / 2 - relax), pc_tr[:, 0] < (l / 2 + relax))
        within_box_y = np.logical_and(pc_tr[:, 1] > (-h - relax), pc_tr[:, 1] < relax)
        within_box_z = np.logical_and(pc_tr[:, 2] > (-w / 2 - relax), pc_tr[:, 2] < (w / 2 + relax))
        within_box = np.logical_and(np.logical_and(within_box_x, within_box_y), within_box_z)

        # Grant segmentation ID (Foreground objects start from 1)
        segm[within_box] = sid + 1
    return segm


parser = argparse.ArgumentParser()
parser.add_argument('data_root', type=str, help='Root path for the dataset')
parser.add_argument('--visualize', dest='visualize', default=False, action='store_true', help='Visualization for check')
args = parser.parse_args()

SRC_DIR = osp.join(args.data_root, 'training')
n_sample = 7481
lidar_root = osp.join(SRC_DIR, "velodyne")
image_root = osp.join(SRC_DIR, "image_2")
calib_root = osp.join(SRC_DIR, "calib")
label_root = osp.join(SRC_DIR, "label_2")

SAVE_DIR = osp.join(args.data_root, 'downsampled')
os.makedirs(SAVE_DIR, exist_ok=True)

clip_distance = 2.0
depth_thresh = 35.0
n_sample_point = 8192

if args.visualize:
    import open3d as o3d
    from utils.visual_util import build_pointcloud, build_bbox3d


pbar = tqdm.tqdm(total=n_sample)
for sid in range(n_sample):
    pc_velo = get_lidar(sid)[:, :3]
    img = get_image(sid)
    calib = get_calibration(sid)
    img_height, img_width, _ = img.shape

    # Extract front-view point cloud from velodyne data
    pts_3d_rect = calib.project_velo_to_rect(pc_velo)
    pts_2d_depth = calib.project_rect_to_image(pts_3d_rect)
    pts_2d, depth = pts_2d_depth[:, :2], pts_2d_depth[:, 2]
    fov_inds = (
            (pts_2d[:, 0] < img_width)
            & (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 1] < img_height)
            & (pts_2d[:, 1] >= 0)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    pc = pts_3d_rect[fov_inds]
    pc[:, :2] *= -1.
    pc = pc.astype(np.float32)
    # Filter by depth
    pc = pc[pc[:, 2] < depth_thresh]

    # FPS downsample
    fps_idx = fps_downsample(pc, n_sample_point=n_sample_point)
    pc = pc[fps_idx]

    # Process bounding box annotations
    boxes_3d = []
    objects = get_label_objects(sid)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d[:, :2] *= -1.
        boxes_3d.append(box3d_pts_3d)
    # Generate segmentation masks from boxes
    segm = box_to_segm(pc, objects)

    # Visualize if needed
    if args.visualize:
        pcds = []
        pcds.append(build_pointcloud(pc, segm, with_background=True))
        if len(boxes_3d) > 0:
            pcds += build_bbox3d(boxes_3d)
        o3d.visualization.draw_geometries(pcds)

    # Save
    save_path = osp.join(SAVE_DIR, '%06d'%(sid))
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, 'pc.npy'), pc)
    np.save(osp.join(save_path, 'segm.npy'), segm)

    pbar.update(1)