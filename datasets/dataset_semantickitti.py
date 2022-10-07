import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

from utils.data_util import compress_label_id, batch_segm_to_mask, augment_transform


class SemanticKITTIDataset(Dataset):
    def __init__(self,
                 data_root,
                 sequence_list=None,
                 decentralize=False,
                 aug_transform=False,
                 aug_transform_args=None,
                 onehot_label=False,
                 max_n_object=10,
                 ignore_npoint_thresh=0):
        """
        :param data_root: root path containing `data'.
        :param mapping_path: a txt file containing the split to be loaded.
        :param decentralize: whether normalize point cloud to be centered at the origin.
        :param aug_transform: whether augment with spatial transformatons.
        :param aug_transform_args: a dict containing hyperparams for sampling spatial augmentations.
        :param onehot_label: whether convert the segmentation to one-hot encoding (only for fully-supervised training).
        :param max_n_object: predefined number of objects per scene that is large enough for the dataset, to be used in one-hot encoding.
        :param ignore_npoint_thresh: threshold to ignore too small objects in GT, only used in supervised training.
        """
        self.data_root = osp.join(data_root, 'downsampled')
        data_ids = sorted(os.listdir(self.data_root))
        if sequence_list is not None:
            self.data_ids = [idx for idx in data_ids if (int(idx[:2]) in sequence_list)]

        self.decentralize = decentralize
        self.aug_transform = aug_transform
        self.aug_transform_args = aug_transform_args
        self.onehot_label = onehot_label
        self.max_n_object = max_n_object
        self.ignore_npoint_thresh = ignore_npoint_thresh


    def __len__(self):
        return len(self.data_ids)


    def _load_data(self, idx):
        data_path = osp.join(self.data_root, self.data_ids[idx])
        pc = np.load(osp.join(data_path, 'pc.npy'))
        segm = np.load(osp.join(data_path, 'segm.npy'))
        return pc, segm


    def __getitem__(self, sid):
        # Extract single-frame point cloud and segmentation
        pc, segm = self._load_data(sid)

        # Normalize point cloud to be centered at the origin
        if self.decentralize:
            center = pc.mean(0)
            pc = pc - center

        # Compress the object-id in segmentation to consecutive numbers starting from 0
        segm = compress_label_id(segm)

        # Make data format consistent with other dynamic dataset
        pcs, segms = np.stack([pc, pc], 0), np.stack([segm, segm], 0)
        flows = np.zeros_like(pcs)

        # Convert the segmentation to one-hot encoding (only for supervised training)
        if self.onehot_label:
            assert self.max_n_object > 0, 'max_n_object must be above 0!'
            segms, valids = batch_segm_to_mask(segms, self.max_n_object, self.ignore_npoint_thresh)
        else:
            valids = np.ones_like(segms, dtype=np.float32)

        # Augment the point cloud & flow with spatial transformations
        if self.aug_transform:
            pcs, flows = augment_transform(pcs, flows, self.aug_transform_args)
            segms = np.concatenate((segms, segms), 0)
            valids = np.concatenate((valids, valids), 0)

        if self.onehot_label:
            return pcs.astype(np.float32), segms.astype(np.float32), flows.astype(np.float32), valids.astype(np.float32)
        else:
            return pcs.astype(np.float32), segms.astype(np.int32), flows.astype(np.float32), valids.astype(np.float32)


    def _save_predsegm(self, mask, save_root, batch_size, n_frame=1, offset=0):
        """
        :param mask: (B, N, K) torch.Tensor.
        """
        mask = mask.detach().cpu().numpy()
        for sid in range(mask.shape[0]):
            segm_pred = mask[sid].argmax(1)
            idx = offset * batch_size + sid
            data_id = self.data_ids[idx]
            save_path = osp.join(save_root, data_id)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, 'segm.npy')
            np.save(save_file, segm_pred)


# Test the dataset loader
if __name__ == '__main__':
    data_root = '/home/ziyang/Desktop/Datasets/SemanticKITTI'
    sequence_list = list(range(11))

    decentralize = True
    aug_transform = True
    aug_transform_args = {
        'scale_low': 0.95,
        'scale_high': 1.05,
        'degree_range': [0, 180, 0],
        'shift_range': [1, 0.1, 1]
    }
    onehot_label = False
    max_n_object = 10
    dataset = SemanticKITTIDataset(data_root=data_root,
                                   sequence_list=sequence_list,
                                   decentralize=decentralize,
                                   aug_transform=aug_transform,
                                   aug_transform_args=aug_transform_args,
                                   onehot_label=onehot_label,
                                   max_n_object=max_n_object)
    print(len(dataset))

    # import random
    # random.shuffle(dataset.data_ids)


    import open3d as o3d
    from utils.visual_util import build_pointcloud

    interval = 50
    for sid in range(len(dataset)):
        pcs, segms, flows, _ = dataset[sid]
        pc1 = pcs[0]
        segm1 = segms[0]
        if onehot_label:
            segm1 = segm1.argmax(1)

        pcds = []
        pcds.append(build_pointcloud(pc1, segm1, with_background=True))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0]))

        # Check spatial augmentations
        if aug_transform:
            pc3 = pcs[2]
            segm3 = segms[2]
            if onehot_label:
                segm3 = segm3.argmax(1)
            pcds.append(build_pointcloud(pc3, segm3, with_background=True).translate([interval, 0.0, 0.0]))
            pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[interval, 0, 0]))

        o3d.visualization.draw_geometries(pcds)