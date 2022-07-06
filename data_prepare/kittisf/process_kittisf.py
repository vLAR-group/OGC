import os
import os.path as osp
import argparse
from multiprocessing import Pool
import numpy as np

from kittisf_util import *


parser = argparse.ArgumentParser()
parser.add_argument('data_root', type=str, help='Root path for the dataset')
args = parser.parse_args()

SRC_DIR = osp.join(args.data_root, 'training')
calib_root = osp.join(SRC_DIR, 'calib_cam_to_cam')
disp1_root = osp.join(SRC_DIR, 'disp_occ_0')
disp2_root = osp.join(SRC_DIR, 'disp_occ_1')
op_flow_root = osp.join(SRC_DIR, 'flow_occ')
instance_map_root = osp.join(SRC_DIR, 'instance')

SAVE_DIR = osp.join(args.data_root, 'processed')
os.makedirs(SAVE_DIR, exist_ok=True)

# Only keep 'Car' & 'Truck' categories
SELECT_SEMANTICS = [26, 28]


def process_one_frame(idx):
    data_id = '%06d'%(idx)

    # Load camera calibration info
    calib_path = osp.join(calib_root, data_id + '.txt')
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)
    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    # Unproject <disparity, optical flow> to <point cloud, scene flow>
    disp1_path = osp.join(disp1_root, data_id + '_10.png')
    disp1, valid_disp1 = load_disp(disp1_path)
    depth1 = disp_2_depth(disp1, valid_disp1, focal_length_pixel)
    pc1 = pixel2xyz(depth1, P_rect_left)

    disp2_path = osp.join(disp2_root, data_id + '_10.png')
    disp2, valid_disp2 = load_disp(disp2_path)
    depth2 = disp_2_depth(disp2, valid_disp2, focal_length_pixel)
    valid_disp = np.logical_and(valid_disp1, valid_disp2)

    op_flow, valid_op_flow = load_op_flow(osp.join(op_flow_root, data_id + '_10.png'))
    vertical = op_flow[..., 1]
    horizontal = op_flow[..., 0]
    height, width = op_flow.shape[:2]

    px2 = np.zeros((height, width), dtype=np.float32)
    py2 = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            if valid_op_flow[i, j] and valid_disp[i, j]:
                try:
                    dx = horizontal[i, j]
                    dy = vertical[i, j]
                except:
                    print('error, i,j:', i, j, 'hor and ver:', horizontal[i, j], vertical[i, j])
                    continue

                px2[i, j] = j + dx
                py2[i, j] = i + dy
    pc2 = pixel2xyz(depth2, P_rect_left, px=px2, py=py2)

    # Load instance segmentation
    instance_map_path = osp.join(instance_map_root, data_id + '_10.png')
    instance_segm = load_segm(instance_map_path)

    # Drop far-away points
    near_mask = np.logical_and(pc1[..., 2] < 35.0, pc2[..., 2] < 35.0)
    final_mask = np.logical_and(np.logical_and(valid_disp, valid_op_flow), near_mask)
    valid_pc1 = pc1[final_mask]
    valid_pc2 = pc2[final_mask]
    valid_instance_segm = instance_segm[final_mask].astype(int)

    # Filter objects according to categories in the segmentation
    valid_instance_segm = filter_segm(valid_instance_segm, select_semantics=SELECT_SEMANTICS)

    # Save
    save_path = osp.join(SAVE_DIR, '%06d'%(idx))
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, 'pc1.npy'), valid_pc1)
    np.save(osp.join(save_path, 'pc2.npy'), valid_pc2)
    np.save(osp.join(save_path, 'segm.npy'), valid_instance_segm)


pool = Pool(4)
indices = range(200)
pool.map(process_one_frame, indices)
pool.close()
pool.join()
