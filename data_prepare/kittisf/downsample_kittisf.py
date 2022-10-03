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
parser.add_argument('data_root', type=str, help='Root path for the dataset')
parser.add_argument('--save_root', type=str, help='Path to save the downsampled dataset')
parser.add_argument('--predflow_path', type=str, default=None, help='Path to load pre-saved flow predictions')
args = parser.parse_args()

data_root = args.data_root
save_root = args.save_root
predflow_path = args.predflow_path

n_sample_point = 8192
SAVE_DIR = osp.join(save_root, 'data')
os.makedirs(SAVE_DIR, exist_ok=True)
if predflow_path is not None:
    SAVE_PREDFLOW_DIR = osp.join(save_root, 'flow_preds', predflow_path)
    os.makedirs(SAVE_PREDFLOW_DIR, exist_ok=True)


# Setup the dataset
from datasets.dataset_kittisf import KITTISceneFlowDataset
mapping_path = 'data_prepare/kittisf/splits/all.txt'
dataset = KITTISceneFlowDataset(data_root=data_root,
                                mapping_path=mapping_path,
                                downsampled=False,
                                predflow_path=predflow_path)

n_scenes = len(dataset)
pbar = tqdm.tqdm(total=n_scenes)
for sid in range(n_scenes):
    pcs, segms, flows = dataset[sid]
    pc_org, segm_org, flow_org = pcs[0], segms[0], flows[0]

    fps_idx = fps_downsample(pc_org, n_sample_point=n_sample_point)
    pc = pc_org[fps_idx]
    segm = segm_org[fps_idx]
    flow = flow_org[fps_idx]

    idx, view_sel_idx = sid // 2, sid % 2
    data_id = dataset.data_ids[idx]
    save_path = osp.join(SAVE_DIR, data_id)
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, 'pc%d.npy'%(view_sel_idx + 1)), pc)
    np.save(osp.join(save_path, 'segm%d.npy'%(view_sel_idx + 1)), segm)

    if predflow_path is not None:
        save_predflow_path = osp.join(SAVE_PREDFLOW_DIR, data_id)
        os.makedirs(save_predflow_path, exist_ok=True)
        np.save(osp.join(save_predflow_path, 'flow%d.npy'%(view_sel_idx + 1)), flow)
    else:
        np.save(osp.join(save_path, 'flow%d.npy'%(view_sel_idx + 1)), flow)

    pbar.update()