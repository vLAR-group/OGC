import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import numpy as np
import glob
import json
from torch.utils.data import Dataset

from utils.data_util import augment_transform


CLASS_NAMES = {'Vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3}


def label_to_onehot(segms, max_n_object):
    """
    Convert the segmentation from object-id to one-hot encoding.
    :param segms: (B, N).
    :param max_n_object: an integer K.
    :return:
        segms_onehot: (B, N, K).
    """
    segms_onehot = []
    for b in range(segms.shape[0]):
        segm = segms[b]
        # (N,) to (N, K)
        _, segm_inv = np.unique(segm, return_inverse=True)
        segm_onehot = np.eye(max_n_object, dtype=np.float32)[segm_inv]
        segms_onehot.append(segm_onehot)
    segms_onehot = np.stack(segms_onehot, 0)
    return segms_onehot


def compress_label_id(segms):
    """
    Compress the object-id in segmentation to consecutive numbers starting from 0 (0, 1, 2, ...).
    :param segm: (B, N).
    :return:
        segm_cpr: (B, N).
    """
    segms_cpr = []
    for b in range(segms.shape[0]):
        segm = segms[b]
        _, segm_cpr = np.unique(segm, return_inverse=True)
        segms_cpr.append(segm_cpr)
    segms_cpr = np.stack(segms_cpr, 0)
    return segms_cpr


class WaymoOpenDataset(Dataset):
    def __init__(self,
                 data_root,
                 mapping_path,
                 downsampled=False,
                 select_frame=None,
                 sampled_interval=1,
                 decentralize=False,
                 aug_transform=False,
                 aug_transform_args=None,
                 onehot_label=False,
                 max_n_object=20,
                 ignore_class_ids=[],
                 ignore_npoint_thresh=0):
        self.data_root = osp.join(data_root, 'data')
        self.sequence_list = [x.strip() for x in open(mapping_path).readlines()]
        self.downsampled = downsampled

        if select_frame is not None:
            print('Loading Waymo dataset. Sample IDs already given in %s'%(select_frame))
            with open(select_frame, 'r') as f:
                data_ids = json.load(f)
                print('Dataset loaded. Number of samples: %d.'%(len(data_ids)))
                self.data_ids = [tuple(data_id) for data_id in data_ids]
        else:
            self.data_ids = self._make_dataset(sampled_interval)

        self.decentralize = decentralize
        self.aug_transform = aug_transform
        self.aug_transform_args = aug_transform_args
        self.onehot_label = onehot_label
        self.max_n_object = max_n_object
        self.ignore_class_ids = ignore_class_ids
        self.ignore_npoint_thresh = ignore_npoint_thresh


    def _make_dataset(self, sampled_interval):
        print('Loading Waymo dataset.')
        data_ids = []
        num_skipped = 0

        for k in range(len(self.sequence_list)):
            sequence_name = osp.splitext(self.sequence_list[k])[0]
            sequence_path = osp.join(self.data_root, sequence_name)
            if not osp.exists(sequence_path):
                num_skipped += 1
                continue

            # Load frames of each sequence
            n_frame = len(glob.glob(osp.join(sequence_path, 'pc_*')))
            for t in range(n_frame):
                data_id = (sequence_name, t)
                data_ids.append(data_id)

        if sampled_interval > 1:
            data_ids = data_ids[::sampled_interval]

        print('Dataset loaded. Total skipped (unavailable) sequences %d/%d'%(num_skipped, len(self.sequence_list)))
        print('Number of samples: %d. Sampled interval: %d.'%(len(data_ids), sampled_interval))
        return data_ids


    def __len__(self):
        return len(self.data_ids)


    def _load_data(self, sequence_name, view_id):
        sequence_path = osp.join(self.data_root, sequence_name)
        pc = np.load(osp.join(sequence_path, 'pc_%04d.npy'%(view_id)))
        segm = np.load(osp.join(sequence_path, 'segm_%04d.npy'%(view_id)))
        semantic_segm = np.load(osp.join(sequence_path, 'semantic_segm_%04d.npy'%(view_id)))
        return [pc], [segm], [semantic_segm]


    def filter_segm(self, segms, semantic_segms):
        segms_filtered, valids = [], []
        for segm, semantic_segm in zip(segms, semantic_segms):
            # Filter out points belonging to specified semantic classes
            ignore_by_class = np.in1d(semantic_segm, self.ignore_class_ids)

            # Filter out points belonging to too small objects
            object_ids, object_sizes = np.unique(segm, return_counts=True)
            object_ids_ignore = object_ids[object_sizes < self.ignore_npoint_thresh]
            ignore_by_size = np.in1d(segm, object_ids_ignore)

            # Combine
            ignore = np.logical_or(ignore_by_class, ignore_by_size)
            # print(ignore_by_class.sum(), ignore_by_size.sum(), ignore.sum())
            segm[ignore] = 0
            segms_filtered.append(segm)
            valid = 1 - ignore.astype(np.int32)
            valids.append(valid)
        return segms_filtered, valids


    def __getitem__(self, sid):
        sequence_name, view_id = self.data_ids[sid]

        # Extract two-frame point cloud, segmentation, and flow
        pcs, segms, semantic_segms = self._load_data(sequence_name, view_id)

        # Filter the segmentation by specified conditions
        segms, valids = self.filter_segm(segms, semantic_segms)

        # If not downsampled, return a list containing two frames with variable number of points
        if not self.downsampled:
            return pcs, segms, valids

        pcs, segms, valids = np.stack(pcs, 0), np.stack(segms, 0), np.stack(valids, 0)

        # Normalize point cloud to be centered at the origin
        if self.decentralize:
            center = pcs.mean(1).mean(0)
            pcs = pcs - center

        # Compress the object-id in segmentation to consecutive numbers starting from 0
        segms = compress_label_id(segms)

        # Convert the segmentation to one-hot encoding (only for supervised training)
        if self.onehot_label:
            assert self.max_n_object > 0, 'max_n_object must be above 0!'
            segms = label_to_onehot(segms, self.max_n_object)
            segms = segms * np.expand_dims(valids, 2)

        # Augment the point cloud with spatial transformations
        if self.aug_transform:
            # Fit an empty flow to fit the function
            pcs = np.concatenate((pcs, pcs), 0)
            flows = np.zeros_like(pcs)
            pcs, _ = augment_transform(pcs, flows, self.aug_transform_args)
            pcs = pcs[[0, 2]]
            segms = np.concatenate((segms, segms), 0)
            valids = np.concatenate((valids, valids), 0)

        if self.onehot_label:
            return pcs.astype(np.float32), segms.astype(np.float32), valids.astype(np.float32)
        else:
            return pcs.astype(np.float32), segms.astype(np.int32), valids.astype(np.int32)


    def _save_predsegm(self, mask, save_root, batch_size, n_frame=1, offset=0):
        """
        :param mask: (B, N, K) torch.Tensor.
        """
        mask = mask.detach().cpu().numpy()
        for sid in range(mask.shape[0]):
            segm_pred = mask[sid].argmax(1)
            idx = offset * batch_size + sid
            sequence_name, view_id = self.data_ids[idx]
            save_path = osp.join(save_root, sequence_name)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, 'segm_%04d.npy'%(view_id))
            np.save(save_file, segm_pred)


if __name__ == '__main__':
    import open3d as o3d
    from utils.visual_util import build_pointcloud
    np.random.seed(0)

    mapping_path = 'data_prepare/waymo/splits/train.txt'
    # mapping_path = 'data_prepare/waymo/splits/val.txt'
    downsampled = True
    if downsampled:
        # data_root = '/home/ziyang/Desktop/Datasets/Waymo_downsampled'
        data_root = '/media/SSD/ziyang/Datasets/Waymo_downsampled'
    else:
        data_root = '/media/SSD/ziyang/Datasets/Waymo'
    select_frame = 'data_prepare/waymo/splits/train_sup.json'    # None
    sampled_interval = 5
    predflow_path = 'flowstep3d_gpf_odo_bound'
    onehot_label = False    # True
    max_n_object = 20
    ignore_class_ids = [2, 3]
    ignore_npoint_thresh = 50
    dataset = WaymoOpenDataset(data_root=data_root,
                               mapping_path=mapping_path,
                               downsampled=downsampled,
                               select_frame=select_frame,
                               sampled_interval=sampled_interval,
                               onehot_label=onehot_label,
                               max_n_object=max_n_object,
                               ignore_class_ids=ignore_class_ids,
                               ignore_npoint_thresh=ignore_npoint_thresh)
    print(len(dataset))

    # Count number of instances per scene
    import tqdm
    n_insts = []
    pbar = tqdm.tqdm(total=len(dataset))
    for sid in range(len(dataset)):
        pcs, segms, _ = dataset[sid]
        segm = segms[0]
        inst_ids = np.unique(segm)
        n_insts.append(inst_ids.shape[0])
        pbar.update()
    n_insts = np.sort(np.array(n_insts))
    print (n_insts.min(), n_insts.mean(), n_insts.max())
    exit()

    """
    After counting, there are maximally
    1) 19 objects in a training sample,
    2) 21 objects in a validation sample.
    (objects with npoint < 50 are ignored)
    """


    np.random.seed(0)
    np.random.shuffle(dataset.data_ids)
    for sid in range(len(dataset)):
        if downsampled:
            pcs, segms, valids = dataset[sid]
        else:
            pcs, segms, valids = dataset[sid]
        pc, segm, valid = pcs[0], segms[0], valids[0]
        print(pc.shape)
        print(np.unique(segm, return_counts=True))
        # print(segm.sum())
        print(valid.mean())

        if sid > 20:
            exit()