import numpy as np
from skspatial.objects import Plane
import torch
from pointnet2.pointnet2 import furthest_point_sample


def fps_downsample(pc, n_sample_point=1024):
    """
    Downsample a point cloud with Furthest Point Sampling (FPS) and return indexes of sampled points.
    :param pc: (N, 3).
    :return:
        fps_idx: (N',).
    """
    pc = torch.from_numpy(pc).unsqueeze(0).cuda().contiguous()
    fps_idx = furthest_point_sample(pc, n_sample_point)
    fps_idx = fps_idx.cpu().numpy()[0]
    return fps_idx


def extract_initial_gpf_seed(pc, n_lpr=20, thresh_seed=0.4, vertical_axis=1):
    """
    :param pc: (N, 3).
    :return:
        seed: (K, 3)
    """
    height = pc[:, vertical_axis]
    lpr = np.partition(height, n_lpr)[:n_lpr].mean()
    seed = pc[height < (lpr + thresh_seed)]
    return seed


def ground_plane_fitting(points, n_sample_point=8192, n_iter=5, n_lpr=200, thresh_seed=0.4, thresh_dist=0.4, vertical_axis=1):
    """
    :param points: (N, 3).
    :param n_sample_point: downsample the point cloud before GPF algorithm for efficiency.
    :return:
        is_ground: (N,).
    """
    if n_sample_point > 0:
        fps_idx = fps_downsample(points, n_sample_point=n_sample_point)
        pc = points[fps_idx]
    else:
        pc = points

    success = False

    while not success:
        # Occasionally, initial seeds are bad and need re-sampling
        try:
            # Extract initial seeds from points with lowest height values
            seed = extract_initial_gpf_seed(pc, n_lpr, thresh_seed, vertical_axis)

            # Iterative plane fitting
            for iter_ in range(n_iter):
                plane = Plane.best_fit(seed)
                center, normal = plane.point, plane.normal
                dist = np.abs(np.einsum('nj,j->n', pc - center, normal))
                is_ground = (dist < thresh_dist)
                seed = pc[is_ground]
            success = True
        except:
            thresh_seed += 0.05

            # Give up if too manny trials fail
            if thresh_seed > 0.8:
                return np.zeros(points.shape[0], dtype=np.int32)

    dist = np.abs(np.einsum('nj,j->n', points - center, normal))
    is_ground = (dist < thresh_dist).astype(np.int32)
    return is_ground


if __name__ == '__main__':
    from datasets.dataset_waymo import WaymoOpenDataset
    import open3d as o3d
    from utils.visual_util import build_pointcloud

    # mapping_path = 'data_prepare/waymo/splits/train.txt'
    mapping_path = 'data_prepare/waymo/splits/val.txt'
    downsampled = False # True
    if downsampled:
        data_root = '/home/ziyang/Desktop/Datasets/Waymo_downsampled'
    else:
        data_root = '/home/ziyang/Desktop/Datasets/Waymo'
    dataset = WaymoOpenDataset(data_root=data_root,
                               mapping_path=mapping_path,
                               downsampled=downsampled)
    sample_ids = list(range(len(dataset)))

    # import random
    # random.shuffle(sample_ids)

    for sid in sample_ids[::50]:
        print(sid)
        if downsampled:
            pcs, segms, flows, _ = dataset[sid]
        else:
            pcs, segms, flows = dataset[sid]
        pc1, pc2 = pcs[0], pcs[1]

        is_ground1 = (pc1[:, 1] < 0.3).astype(np.int32)
        is_ground1_gpf = ground_plane_fitting(pc1, n_sample_point=2048, n_lpr=50)
        is_ground1_joint = np.maximum(is_ground1, is_ground1_gpf)
        pcds = []
        interval = 60
        pcds.append(build_pointcloud(pc1, 1 - is_ground1, with_background=True))
        pcds.append(build_pointcloud(pc1, 1 - is_ground1_gpf, with_background=True).translate([interval, 0, 0]))
        pcds.append(build_pointcloud(pc1, 1 - is_ground1_joint, with_background=True).translate([2 * interval, 0, 0]))
        o3d.visualization.draw_geometries(pcds)