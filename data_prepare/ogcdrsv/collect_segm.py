import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import shutil

import open3d as o3d

from utils.data_util import fps_downsample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str,
                        default='/home/ziyang/Desktop/Datasets/OGC_DynamicRoom', help='Root path for the OGC-DR dataset')
    parser.add_argument('--dest_root', type=str,
                        default='/home/ziyang/Desktop/Datasets/OGC_DynamicRoom_SingleView', help='Root path for the OGC-DRSV dataset')
    args = parser.parse_args()

    src_root = osp.join(args.src_root, 'data')
    data_root = osp.join(args.dest_root, 'pcd')
    data_ids = sorted(os.listdir(data_root))
    n_frame = 4
    n_sample_point = 2048
    save_root = osp.join(args.dest_root, 'data')
    os.makedirs(save_root, exist_ok=True)


    pbar = tqdm(total=len(data_ids))
    for data_id in data_ids:
        n_object = int(data_id[:2])
        data_path = osp.join(data_root, data_id)
        src_path = osp.join(src_root, data_id)
        save_path = osp.join(save_root, data_id)
        os.makedirs(save_path, exist_ok=True)

        for frame_id in range(n_frame):
            pcd_file = osp.join(data_path, 'pc_%02d.pcd'%(frame_id))
            pcd = o3d.io.read_point_cloud(pcd_file)
            pc = np.asarray(pcd.points).astype(np.float32)

            # Downsample
            fps_idx = fps_downsample(pc, n_sample_point=n_sample_point)
            pc = pc[fps_idx]

            # Load complete point cloud and segmentation
            pc_src = np.load(osp.join(src_path, 'pc_%02d.npy'%(frame_id)))
            segm_src = np.load(osp.join(src_path, 'segm_%02d.npy'%(frame_id)))
            pose = np.load(osp.join(src_path, 'pose_%02d.npy'%(frame_id)))

            # Interpolate
            dist = cdist(pc, pc_src)
            nearest = dist.argmin(1)
            segm = segm_src[nearest]

            np.save(osp.join(save_path, 'pc_%02d.npy'%(frame_id)), pc)
            np.save(osp.join(save_path, 'segm_%02d.npy'%(frame_id)), segm)
            np.save(osp.join(save_path, 'pose_%02d.npy'%(frame_id)), pose)

            # from utils.visual_util import build_pointcloud
            # pcds = [build_pointcloud(pc, segm)]
            # o3d.visualization.draw_geometries(pcds)

        pbar.update(1)

    pbar.close()

    # Copy the train/val/test split file
    for split in ['train', 'val', 'test']:
        shutil.copyfile(osp.join(src_root, split+'.lst'), osp.join(save_root, split+'.lst'))