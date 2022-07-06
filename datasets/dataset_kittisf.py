import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

from utils.data_util import compress_label_id, label_to_onehot, augment_transform


class KITTISceneFlowDataset(Dataset):
    def __init__(self,
                 data_root,
                 mapping_path,
                 downsampled=False,
                 view_sels=[[0, 1]],
                 predflow_path=None,
                 decentralize=False,
                 aug_transform=False,
                 aug_transform_args=None,
                 onehot_label=False,
                 max_n_object=10,
                 ignore_npoint_thresh=0):
        """
        :param data_root: root path containing `data'.
        :param mapping_path: a txt file containing the split to be loaded.
        :param downsampled: whether to load a downsampled version or the original version (variable number of points per scene).
        :param view_sels: paired combinations of views to be used.
        :param predflow_path: path to load pre-saved flow predictions, otherwise use GT flows.
        :param decentralize: whether normalize point cloud to be centered at the origin.
        :param aug_transform: whether augment with spatial transformatons.
        :param aug_transform_args: a dict containing hyperparams for sampling spatial augmentations.
        :param onehot_label: whether convert the segmentation to one-hot encoding (only for fully-supervised training).
        :param max_n_object: predefined number of objects per scene that is large enough for the dataset, to be used in one-hot encoding.
        :param ignore_npoint_thresh: threshold to ignore too small objects in GT, only used in supervised training.
        """
        if downsampled:
            self.data_root = osp.join(data_root, 'data')
        else:
            self.data_root = osp.join(data_root, 'processed')

        with open(mapping_path, 'r') as f:
            self.data_ids = f.read().strip().split('\n')
        self.view_sels = view_sels

        if predflow_path is not None:
            self.predflow_path = osp.join(data_root, 'flow_preds', predflow_path)
        else:
            self.predflow_path = None

        self.downsampled = downsampled
        self.decentralize = decentralize
        self.aug_transform = aug_transform
        self.aug_transform_args = aug_transform_args
        self.onehot_label = onehot_label
        self.max_n_object = max_n_object
        self.ignore_npoint_thresh = ignore_npoint_thresh


    def __len__(self):
        return len(self.data_ids) * len(self.view_sels)


    def _load_data(self, idx, view_sel):
        data_path = osp.join(self.data_root, self.data_ids[idx])

        view_id1, view_id2 = view_sel
        pc1, pc2 = np.load(osp.join(data_path, 'pc%d.npy'%(view_id1 + 1))), np.load(osp.join(data_path, 'pc%d.npy'%(view_id2 + 1)))
        if self.downsampled:
            segm1, segm2 = np.load(osp.join(data_path, 'segm%d.npy'%(view_id1 + 1))), np.load(osp.join(data_path, 'segm%d.npy'%(view_id2 + 1)))
            flow1, flow2 = np.load(osp.join(data_path, 'flow%d.npy'%(view_id1 + 1))), np.load(osp.join(data_path, 'flow%d.npy'%(view_id2 + 1)))
        else:
            segm = np.load(osp.join(data_path, 'segm.npy'))
            segm1, segm2 = segm, segm
            flow1, flow2 = (pc2 - pc1), (pc1 - pc2)

        return [pc1, pc2], [segm1, segm2], [flow1, flow2]


    def _load_predflow(self, idx, view_sel):
        data_path = osp.join(self.predflow_path, self.data_ids[idx])
        view_id1, view_id2 = view_sel
        flow1, flow2 = np.load(osp.join(data_path, 'flow%d.npy'%(view_id1 + 1))), np.load(osp.join(data_path, 'flow%d.npy'%(view_id2 + 1)))
        return [flow1, flow2]


    def __getitem__(self, sid):
        idx, view_sel_idx = sid // len(self.view_sels), sid % len(self.view_sels)
        view_sel = self.view_sels[view_sel_idx]

        # Extract two-frame point cloud, segmentation, and flow
        pcs, segms, flows = self._load_data(idx, view_sel)
        if self.predflow_path is not None:
            flows = self._load_predflow(idx, view_sel)
        pcs, segms, flows = np.stack(pcs, 0), np.stack(segms, 0), np.stack(flows, 0)

        # Normalize point cloud to be centered at the origin
        if self.decentralize:
            center = pcs.mean(1).mean(0)
            pcs = pcs - center

        # Compress the object-id in segmentation to consecutive numbers starting from 0
        segms = np.reshape(segms, -1)
        segms = compress_label_id(segms)
        segms = np.reshape(segms, (2, -1))

        # Convert the segmentation to one-hot encoding (only for supervised training)
        if self.onehot_label:
            assert self.max_n_object > 0, 'max_n_object must be above 0!'
            segms, valids = label_to_onehot(segms, self.max_n_object, self.ignore_npoint_thresh)
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


    def _save_predflow(self, flow_pred, save_root, batch_size, n_frame=1, offset=0):
        """
        :param flow_pred: (B, N, 3) torch.Tensor.
        """
        flow_pred = flow_pred.detach().cpu().numpy()
        for sid in range(flow_pred.shape[0]):
            save_flow = flow_pred[sid]
            idx, view_sel_idx = (offset * batch_size + sid) // n_frame, (offset * batch_size + sid) % n_frame
            data_id = self.data_ids[idx]
            save_path = osp.join(save_root, data_id)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, 'flow%d.npy'%(view_sel_idx + 1))
            np.save(save_file, save_flow)


    def _save_predsegm(self, mask, save_root, batch_size, n_frame=1, offset=0):
        """
        :param mask: (B, N, K) torch.Tensor.
        """
        mask = mask.detach().cpu().numpy()
        for sid in range(mask.shape[0]):
            segm_pred = mask[sid].argmax(1)
            idx, view_sel_idx = (offset * batch_size + sid) // n_frame, (offset * batch_size + sid) % n_frame
            data_id = self.data_ids[idx]
            save_path = osp.join(save_root, data_id)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, 'segm%d.npy'%(view_sel_idx + 1))
            np.save(save_file, segm_pred)


# Test the dataset loader
if __name__ == '__main__':
    mapping_path = 'data_prepare/kittisf/splits/all.txt'
    downsampled = True
    if downsampled:
        data_root = '/home/ziyang/Desktop/Datasets/KITTI_SceneFlow_downsampled'
    else:
        data_root = '/home/ziyang/Desktop/Datasets/KITTI_SceneFlow'
    view_sels = [[0, 1]]
    predflow_path = 'flowstep3d'

    decentralize = False
    aug_transform = False
    aug_transform_args = {
        'scale_low': 0.95,
        'scale_high': 1.05,
        'degree_range': [0, 180, 0],
        'shift_range': [1, 0.1, 1]
    }
    onehot_label = False
    max_n_object = 10
    dataset = KITTISceneFlowDataset(data_root=data_root,
                                    mapping_path=mapping_path,
                                    downsampled=downsampled,
                                    view_sels=view_sels,
                                    predflow_path=predflow_path,
                                    decentralize=decentralize,
                                    aug_transform=aug_transform,
                                    aug_transform_args=aug_transform_args,
                                    onehot_label=onehot_label,
                                    max_n_object=max_n_object)


    import open3d as o3d
    from utils.visual_util import build_pointcloud

    interval = 50
    for sid in range(len(dataset)):
        pcs, segms, flows, _ = dataset[sid]
        pc1, pc2 = pcs[0], pcs[1]
        segm1, segm2 = segms[0], segms[1]
        if onehot_label:
            segm1, segm2 = segm1.argmax(1), segm2.argmax(1)

        pcds = []
        pcds.append(build_pointcloud(pc1, segm1, with_background=True))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0]))
        pcds.append(build_pointcloud(pc2, segm2, with_background=True).translate([interval, 0.0, 0.0]))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[interval, 0, 0]))

        # Check spatial augmentations
        if aug_transform:
            pc3, pc4 = pcs[2], pcs[3]
            segm3, segm4 = segms[2], segms[3]
            if onehot_label:
                segm3, segm4 = segm3.argmax(1), segm4.argmax(1)
            pcds.append(build_pointcloud(pc3, segm3, with_background=True).translate([2 * interval, 0.0, 0.0]))
            pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[2 * interval, 0, 0]))
            pcds.append(build_pointcloud(pc4, segm4, with_background=True).translate([3 * interval, 0.0, 0.0]))
            pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[3 * interval, 0, 0]))

        # Check flows
        flow1, flow2 = flows[0], flows[1]
        pcds.append(build_pointcloud(pc1, np.zeros_like(segm1)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc1 + flow1, np.zeros_like(segm1)).translate([-1 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-1 * interval, 0.0, 0.0]))

        o3d.visualization.draw_geometries(pcds)