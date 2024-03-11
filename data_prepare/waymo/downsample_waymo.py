import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import tqdm
import argparse
import numpy as np

from utils.data_util import fps_downsample


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, help='Root path for the dataset')
parser.add_argument('--save_root', type=str, help='Path to save the downsampled dataset')
parser.add_argument('--split', type=str, default='train', help='Dataset split to process (train/val)')
parser.add_argument('--predflow_path', type=str, default=None, help='Path to load pre-saved flow predictions')
args = parser.parse_args()

data_root = args.data_root
predflow_path = args.predflow_path
n_sample_point = 8192

SAVE_DIR = osp.join(args.save_root, 'data')
os.makedirs(SAVE_DIR, exist_ok=True)
if predflow_path is not None:
    predflow_root = osp.join(data_root, 'flow_preds', predflow_path)
    SAVE_PREDFLOW_DIR = osp.join(args.save_root, 'flow_preds', predflow_path)
    os.makedirs(SAVE_PREDFLOW_DIR, exist_ok=True)


# Setup the dataset (Only for collecting sample indexes)
from datasets.dataset_waymo import WaymoOpenDataset
if args.split == 'val':
    mapping_path = 'data_prepare/waymo/splits/val.txt'
else:
    mapping_path = 'data_prepare/waymo/splits/train.txt'
dataset = WaymoOpenDataset(data_root=data_root,
                           mapping_path=mapping_path)


n_scenes = len(dataset)
pbar = tqdm.tqdm(total=n_scenes)
for sid in range(n_scenes):
    sequence_name, view_id1, view_id2 = dataset.data_ids[sid]
    sequence_path = osp.join(data_root, 'data', sequence_name)

    pc1_org, pc2_org = np.load(osp.join(sequence_path, 'pc_%04d.npy'%(view_id1))), np.load(osp.join(sequence_path, 'pc_%04d.npy'%(view_id2)))
    segm1_org, segm2_org = np.load(osp.join(sequence_path, 'segm_%04d.npy'%(view_id1))), np.load(osp.join(sequence_path, 'segm_%04d.npy'%(view_id2)))
    semantic_segm1_org, semantic_segm2_org = \
        np.load(osp.join(sequence_path, 'semantic_segm_%04d.npy'%(view_id1))), np.load(osp.join(sequence_path, 'semantic_segm_%04d.npy'%(view_id2)))
    if predflow_path is not None:
        flow_org = np.load(osp.join(predflow_root, sequence_name, 'flow_%04d_%04d.npy'%(view_id1, view_id2)))
    else:
        flow_org = np.load(osp.join(sequence_path, 'flow_%04d_%04d.npy'%(view_id1, view_id2)))

    if pc1_org.shape[0] > 0:     # Occasionally, no points
        fps_idx1 = fps_downsample(pc1_org, n_sample_point=n_sample_point)
        pc1, segm1, semantic_segm1 = pc1_org[fps_idx1], segm1_org[fps_idx1], semantic_segm1_org[fps_idx1]
        flow = flow_org[fps_idx1]
    else:
        pc1, segm1, semantic_segm1 = pc1_org, segm1_org, semantic_segm1_org
        flow = flow_org

    if view_id2 == 0:
        # "view_id2 > 0" means this frame has been processed as "view_id1" in the last frame pair
        if pc2_org.shape[0] > 0:
            fps_idx2 = fps_downsample(pc2_org, n_sample_point=n_sample_point)
            pc2, segm2, semantic_segm2 = pc2_org[fps_idx2], segm2_org[fps_idx2], semantic_segm2_org[fps_idx2]
        else:
            pc2, segm2, semantic_segm2 = pc2_org, segm2_org, semantic_segm2_org

    save_path = osp.join(SAVE_DIR, sequence_name)
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, 'pc_%04d.npy' % (view_id1)), pc1)
    np.save(osp.join(save_path, 'segm_%04d.npy'%(view_id1)), segm1)
    np.save(osp.join(save_path, 'semantic_segm_%04d.npy' % (view_id1)), semantic_segm1)
    if view_id2 == 0:
        np.save(osp.join(save_path, 'pc_%04d.npy' % (view_id2)), pc2)
        np.save(osp.join(save_path, 'segm_%04d.npy'%(view_id2)), segm2)
        np.save(osp.join(save_path, 'semantic_segm_%04d.npy' % (view_id2)), semantic_segm2)

    if predflow_path is not None:
        save_predflow_path = osp.join(SAVE_PREDFLOW_DIR, sequence_name)
        os.makedirs(save_predflow_path, exist_ok=True)
        np.save(osp.join(save_predflow_path, 'flow_%04d_%04d.npy' % (view_id1, view_id2)), flow)
    else:
        np.save(osp.join(save_path, 'flow_%04d_%04d.npy' % (view_id1, view_id2)), flow)

    pbar.update()