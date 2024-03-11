import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import tqdm
import numpy as np
import json


def convert_id_to_pair(data_ids):
    paired_data_ids = []
    for data_id in data_ids:
        sequence_name, view_id = data_id
        if view_id > 0:
            view_id1, view_id2 = view_id, view_id - 1
            paired_data_ids.append((sequence_name, view_id1, view_id2))
    return paired_data_ids


def detect_moving(pc, flow, bg_rot, bg_transl, thresh=0.2):
    # Count the number of moving points
    flow_fitted = np.einsum('ij,nj->ni', bg_rot, pc) + bg_transl - pc
    diff = np.linalg.norm(flow_fitted - flow, axis=1)
    n_mov_point = (diff > thresh).astype(np.float32).sum()
    return n_mov_point


if __name__ == '__main__':
    from datasets.dataset_waymo import WaymoOpenDataset
    split = 'train'
    # split = 'val'

    # Convert single-frame data IDs to frame-pair data IDs
    select_frame = 'data_prepare/waymo/splits/%s_sup.json' % (split)
    with open(select_frame, 'r') as f:
        data_ids = json.load(f)
    data_ids = convert_id_to_pair(data_ids)
    save_select_frame = 'data_prepare/waymo/splits/%s_sup_paired.json' % (split)
    with open(save_select_frame, 'w') as f:
        json.dump(data_ids, f)

    # Setup the dataset
    DATA_ROOT = "/media/SSD/ziyang/Datasets/Waymo_downsampled"
    if split == 'val':
        mapping_path = 'data_prepare/waymo/splits/val.txt'
    else:
        mapping_path = 'data_prepare/waymo/splits/train.txt'
    ignore_class_ids = [2, 3]   # Ignore Pedestrian & Cyclist
    ignore_npoint_thresh = 50
    select_frame = save_select_frame
    predflow_path = 'flowstep3d_gpf_odo_bound'
    dataset = WaymoOpenDataset(data_root=DATA_ROOT,
                               mapping_path=mapping_path,
                               downsampled=True,
                               select_frame=select_frame,
                               predflow_path=predflow_path,
                               ignore_class_ids=ignore_class_ids,
                               ignore_npoint_thresh=ignore_npoint_thresh)
    moving_thresh = 0.2
    n_mov_point_ratio_thresh = 0.2


    moving_samples = []
    sample_ids = list(range(len(dataset)))
    pbar = tqdm.tqdm(total=len(sample_ids))
    for sid in sample_ids:
        sequence_name, view_id1, view_id2 = dataset.data_ids[sid]
        pbar.update()
        # print (sequence_name, view_id1, view_id2)

        pcs, segms, flows, _ = dataset[sid]
        pc, segm, flow = pcs[0], segms[0], flows[0]

        # Exclude samples with pure background
        if np.unique(segm).shape[0] == 1:
            continue

        # Load groundtruth ego-motion
        sequence_path = osp.join("/media/SSD/ziyang/Datasets/Waymo", 'data', sequence_name)
        pose1, pose2 = np.load(osp.join(sequence_path, 'pose_%04d.npy' % (view_id1))), np.load(
            osp.join(sequence_path, 'pose_%04d.npy' % (view_id2)))
        rot1, transl1 = pose1[0:3, 0:3], pose1[0:3, 3]
        rot2, transl2 = pose2[0:3, 0:3], pose2[0:3, 3]
        rot = rot2.T @ rot1
        transl = rot2.T @ (transl1 - transl2)

        # Check the number of moving objects
        not_ground = (pc[:, 1] >= 0.3)
        pc_fg, segm_fg, flow_fg = pc[not_ground], segm[not_ground], flow[not_ground]
        n_mov_point = detect_moving(pc_fg, flow_fg, rot, transl, thresh=moving_thresh)
        n_mov_point_ratio = n_mov_point / pc_fg.shape[0]

        if (n_mov_point_ratio > n_mov_point_ratio_thresh):
            moving_samples.append((sequence_name, view_id1, view_id2))

    print(len(dataset), len(moving_samples))
    save_file = 'data_prepare/waymo/splits/%s_unsup.json'%(split)
    with open(save_file, 'w') as f:
        json.dump(moving_samples, f)