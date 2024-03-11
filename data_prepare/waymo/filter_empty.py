import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import json
import tqdm

from datasets.dataset_waymo_singleframe import WaymoOpenDataset


# Select samples for fully-supervised single-frame segmentation
split = 'train'
# split = 'val'
if split == 'val':
    mapping_path = 'data_prepare/waymo/splits/val.txt'
else:
    mapping_path = 'data_prepare/waymo/splits/train.txt'
downsampled = False
if downsampled:
    data_root = '/media/SSD/ziyang/Datasets/Waymo_downsampled'
else:
    data_root = '/media/SSD/ziyang/Datasets/Waymo'
sampled_interval = 5
dataset = WaymoOpenDataset(data_root=data_root,
                           mapping_path=mapping_path,
                           downsampled=downsampled,
                           sampled_interval=sampled_interval)

pbar = tqdm.tqdm(total=len(dataset))
keep_samples = []
ignore_samples = []
for sid in range(len(dataset)):
    pcs, segms, flows = dataset[sid]
    sequence_name, view_id = dataset.data_ids[sid]
    pc = pcs[0]
    if pc.shape[0] < 8192:
        print(sequence_name, view_id, pc.shape[0])
        ignore_samples.append((sequence_name, view_id))
    else:
        keep_samples.append((sequence_name, view_id))
    pbar.update()

save_file = 'data_prepare/waymo/splits/%s_sup.json'%(split)
with open(save_file, 'w') as f:
    json.dump(keep_samples, f)