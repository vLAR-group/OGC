import os
import os.path as osp
import json
import numpy as np
from torch.utils.data import Dataset

from utils.data_util import compress_label_id, label_to_onehot, augment_transform


def compute_flow(pc1, segm1, pose1, pose2):
    """
    Compute per-point scene flows from object pose changes.
    :param pc1: (N, 3)
    :param segm1: (N, 3)
    :param pose1 & pose2: (K, 4, 4)
    :return:
        flow: (N, 3)
    """
    flow = np.zeros_like(pc1)
    n_object = pose1.shape[0]
    for k in range(n_object):
        rel_pose = np.matmul(pose2[k], np.linalg.inv(pose1[k]))
        rot, transl = rel_pose[:3, :3], rel_pose[:3, 3]
        mask_k = (segm1 == (k+1))       # foreground object id start from 1 in OGC-DR
        flow_k = np.einsum('ij,nj->ni', rot, pc1[mask_k]) + transl - pc1[mask_k]
        flow[mask_k] = flow_k
    return flow


class OGCDynamicRoomDataset(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 view_sels=[[0, 1]],
                 predflow_path=None,
                 decentralize=False,
                 aug_transform=False,
                 aug_transform_args=None,
                 onehot_label=False,
                 max_n_object=8):
        """
        :param data_root: root path containing `data'.
        :param split: split to be loaded.
        :param view_sels: paired combinations of views to be used.
        :param predflow_path: path to load pre-saved flow predictions, otherwise use GT flows.
        :param decentralize: whether normalize point cloud to be centered at the origin.
        :param aug_transform: whether augment with spatial transformatons.
        :param aug_transform_args: a dict containing hyperparams for sampling spatial augmentations.
        :param onehot_label: whether convert the segmentation to one-hot encoding (only for fully-supervised training).
        :param max_n_object: predefined number of objects per scene that is large enough for the dataset, to be used in one-hot encoding.
        """
        self.data_root = osp.join(data_root, 'data')
        self.split = split
        with open(osp.join(self.data_root, split+'.lst'), 'r') as f:
            self.data_ids = f.read().strip().split('\n')
        self.view_sels = view_sels

        if predflow_path is not None:
            self.predflow_path = osp.join(data_root, 'flow_preds', predflow_path)
            pf_meta = self.predflow_path + '.json'
            with open(pf_meta, 'r') as f:
                self.pf_view_sels = json.load(f)['view_sel']
            # Check if flow predictions cover the specified "view_sel"
            if any([sel not in self.pf_view_sels for sel in view_sels]):
                raise ValueError('Flow predictions cannot cover specified view selections!')
        else:
            self.predflow_path = None

        self.decentralize = decentralize
        self.aug_transform = aug_transform
        self.aug_transform_args = aug_transform_args
        self.onehot_label = onehot_label
        self.max_n_object = max_n_object


    def __len__(self):
        return len(self.data_ids) * len(self.view_sels)


    def _load_data(self, idx, view_sel):
        data_path = osp.join(self.data_root, self.data_ids[idx])

        pcs, segms, poses = [], [], []
        for view in view_sel:
            pc = np.load(osp.join(data_path, 'pc_%02d.npy'%(view)))
            pcs.append(pc)
            segm = np.load(osp.join(data_path, 'segm_%02d.npy'%(view)))
            segms.append(segm)
            pose = np.load(osp.join(data_path, 'pose_%02d.npy'%(view)))
            poses.append(pose)
        return pcs, segms, poses


    def _load_predflow(self, idx):
        data_path = osp.join(self.predflow_path, self.data_ids[idx] + '.npy')
        flow_pred = np.load(data_path)
        return flow_pred


    def __getitem__(self, sid):
        idx, view_sel_idx = sid // len(self.view_sels), sid % len(self.view_sels)
        view_sel = self.view_sels[view_sel_idx]

        # Extract two-frame point cloud, segmentation, and flow
        pcs, segms, poses = self._load_data(idx, view_sel)
        flows = []
        if self.predflow_path is not None:
            view_id1, view_id2 = view_sel
            flow_pred = self._load_predflow(idx)
            flows.append(flow_pred[self.pf_view_sels.index([view_id1, view_id2])])
            flows.append(flow_pred[self.pf_view_sels.index([view_id2, view_id1])])
        else:
            flows.append(compute_flow(pcs[0], segms[0], poses[0], poses[1]))
            flows.append(compute_flow(pcs[1], segms[1], poses[1], poses[0]))
        pcs, segms, flows = np.stack(pcs, 0), np.stack(segms, 0), np.stack(flows, 0)

        # Normalize point cloud to be centered at the origin
        if self.decentralize:
            center = pcs.mean(1).mean(0)
            pcs = pcs - center

        # Compress the object-id in segmentation to consecutive numbers starting from 0
        segms = np.reshape(segms, -1)
        segms = compress_label_id(segms)
        segms = np.reshape(segms, (2, -1))

        # Convert the segmentation to one-hot encoding (only for fully-supervised training)
        if self.onehot_label:
            assert self.max_n_object > 0, 'max_n_object must be above 0!'
            segms, valids = label_to_onehot(segms, self.max_n_object, ignore_npoint_thresh=0)
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
        for sid in range(flow_pred.shape[0] // n_frame):
            save_flow = flow_pred[sid * n_frame:(sid + 1) * n_frame]
            idx = offset * batch_size // n_frame + sid
            data_id = self.data_ids[idx]
            save_file = osp.join(save_root, data_id + '.npy')
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
            save_file = os.path.join(save_path, 'segm_%02d.npy'%(view_sel_idx))
            np.save(save_file, segm_pred)


# Test the dataset loader
if __name__ == '__main__':
    split = 'test'
    data_root = '/home/ziyang/Desktop/Datasets/OGC_DynamicRoom'
    view_sels = [[0, 1]]
    predflow_path = None

    decentralize = True
    aug_transform = True
    aug_transform_args = {
        'scale_low': 0.95,
        'scale_high': 1.05,
        'degree_range': [0, 10, 0],
        'shift_range': [0.05, 0.05, 0.05],
        'aug_pc2':{
            'degree_range': [0, 0, 0],
            'shift_range': [0.01, 0.01, 0.01]
        }
    }
    onehot_label = False
    max_n_object = 8
    dataset = OGCDynamicRoomDataset(data_root=data_root,
                                    split=split,
                                    view_sels=view_sels,
                                    predflow_path=predflow_path,
                                    decentralize=decentralize,
                                    aug_transform=aug_transform,
                                    aug_transform_args=aug_transform_args,
                                    onehot_label=onehot_label,
                                    max_n_object=max_n_object)
    print (len(dataset))


    import open3d as o3d
    from utils.visual_util import build_pointcloud

    interval = 1.2
    for sid in range(len(dataset)):
        pcs, segms, flows, _ = dataset[sid]
        pc1, pc2 = pcs[0], pcs[1]
        segm1, segm2 = segms[0], segms[1]
        if onehot_label:
            segm1, segm2 = segm1.argmax(1), segm2.argmax(1)

        pcds = []
        pcds.append(build_pointcloud(pc1, segm1))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))
        pcds.append(build_pointcloud(pc2, segm2).translate([interval, 0.0, 0.0]))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[interval, 0, 0]))

        # Check spatial augmentations
        if aug_transform:
            pc3, pc4 = pcs[2], pcs[3]
            segm3, segm4 = segms[2], segms[3]
            if onehot_label:
                segm3, segm4 = segm3.argmax(1), segm4.argmax(1)
            pcds.append(build_pointcloud(pc3, segm3).translate([2 * interval, 0.0, 0.0]))
            pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[2 * interval, 0, 0]))
            pcds.append(build_pointcloud(pc4, segm4).translate([3 * interval, 0.0, 0.0]))
            pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[3 * interval, 0, 0]))

        # Check flows
        flow1, flow2 = flows[0], flows[1]
        pcds.append(build_pointcloud(pc1, np.zeros_like(segm1)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc1 + flow1, np.zeros_like(segm1)).translate([-1 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-1 * interval, 0.0, 0.0]))

        o3d.visualization.draw_geometries(pcds)