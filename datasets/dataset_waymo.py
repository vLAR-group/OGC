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

from utils.data_util import compress_label_id, augment_transform


CLASS_NAMES = {'Vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3}


class WaymoOpenDataset(Dataset):
    def __init__(self,
                 data_root,
                 mapping_path,
                 downsampled=False,
                 select_frame=None,
                 sampled_interval=1,
                 predflow_path=None,
                 decentralize=False,
                 aug_transform=False,
                 aug_transform_args=None,
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

        if predflow_path is not None:
            self.predflow_path = osp.join(data_root, 'flow_preds', predflow_path)
        else:
            self.predflow_path = None

        self.decentralize = decentralize
        self.aug_transform = aug_transform
        self.aug_transform_args = aug_transform_args
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
            for t in range(1, n_frame):
                # Waymo only contains backward scene flow
                view_id1, view_id2 = t, t - 1
                data_id = (sequence_name, view_id1, view_id2)
                data_ids.append(data_id)

        if sampled_interval > 1:
            data_ids = data_ids[::sampled_interval]

        print('Dataset loaded. Total skipped (unavailable) sequences %d/%d'%(num_skipped, len(self.sequence_list)))
        print('Number of samples: %d. Sampled interval: %d.'%(len(data_ids), sampled_interval))
        return data_ids


    def __len__(self):
        return len(self.data_ids)


    def _load_data(self, sequence_name, view_id1, view_id2):
        sequence_path = osp.join(self.data_root, sequence_name)
        pc1, pc2 = np.load(osp.join(sequence_path, 'pc_%04d.npy'%(view_id1))), np.load(osp.join(sequence_path, 'pc_%04d.npy'%(view_id2)))
        segm1, segm2 = np.load(osp.join(sequence_path, 'segm_%04d.npy'%(view_id1))), np.load(osp.join(sequence_path, 'segm_%04d.npy'%(view_id2)))
        semantic_segm1 = np.load(osp.join(sequence_path, 'semantic_segm_%04d.npy'%(view_id1)))
        semantic_segm2 = np.load(osp.join(sequence_path, 'semantic_segm_%04d.npy'%(view_id2)))
        return [pc1, pc2], [segm1, segm2], [semantic_segm1, semantic_segm2]


    def _load_flow(self, sequence_name, view_id1, view_id2):
        sequence_path = osp.join(self.data_root, sequence_name)
        flow = np.load(osp.join(sequence_path, 'flow_%04d_%04d.npy'%(view_id1, view_id2)))
        return [flow, flow]


    def _load_predflow(self, sequence_name, view_id1, view_id2):
        sequence_path = osp.join(self.predflow_path, sequence_name)
        flow = np.load(osp.join(sequence_path, 'flow_%04d_%04d.npy'%(view_id1, view_id2)))
        return [flow, flow]


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
        sequence_name, view_id1, view_id2 = self.data_ids[sid]

        # Extract two-frame point cloud, segmentation, and flow
        pcs, segms, semantic_segms = self._load_data(sequence_name, view_id1, view_id2)
        if self.predflow_path is not None:
            flows = self._load_predflow(sequence_name, view_id1, view_id2)
        else:
            flows = self._load_flow(sequence_name, view_id1, view_id2)

        # Filter the segmentation by specified conditions
        segms, valids = self.filter_segm(segms, semantic_segms)

        # If not downsampled, return a list containing two frames with variable number of points
        if not self.downsampled:
            return pcs, segms, flows, valids

        pcs, segms, flows, valids = np.stack(pcs, 0), np.stack(segms, 0), np.stack(flows, 0), np.stack(valids, 0)

        # Normalize point cloud to be centered at the origin
        if self.decentralize:
            center = pcs.mean(1).mean(0)
            pcs = pcs - center

        # Compress the object-id in segmentation to consecutive numbers starting from 0
        segms = np.reshape(segms, -1)
        segms = compress_label_id(segms)
        segms = np.reshape(segms, (2, -1))

        # Augment the point cloud & flow with spatial transformations
        if self.aug_transform:
            pcs, flows = augment_transform(pcs, flows, self.aug_transform_args)
            segms = np.concatenate((segms, segms), 0)
            valids = np.concatenate((valids, valids), 0)

        return pcs.astype(np.float32), segms.astype(np.int32), flows.astype(np.float32), valids.astype(np.float32)


    def _save_predflow(self, flow_pred, save_root, batch_size, n_frame=1, offset=0):
        """
        :param flow_pred: (B, N, 3) torch.Tensor.
        """
        flow_pred = flow_pred.detach().cpu().numpy()
        for sid in range(flow_pred.shape[0]):
            save_flow = flow_pred[sid]
            idx = (offset * batch_size + sid) // n_frame
            sequence_name, view_id1, view_id2 = self.data_ids[idx]
            save_path = osp.join(save_root, sequence_name)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, 'flow_%04d_%04d.npy'%(view_id1, view_id2))
            np.save(save_file, save_flow)


if __name__ == '__main__':
    import open3d as o3d
    from utils.visual_util import build_pointcloud

    mapping_path = 'data_prepare/waymo/splits/train.txt'
    # mapping_path = 'data_prepare/waymo/splits/val.txt'
    downsampled = True
    if downsampled:
        data_root = '/media/SSD/ziyang/Datasets/Waymo_downsampled'
    else:
        data_root = '/media/SSD/ziyang/Datasets/Waymo'
    dataset = WaymoOpenDataset(data_root=data_root,
                               mapping_path=mapping_path,
                               downsampled=downsampled)
    print(len(dataset))

    # # Count number of instances per scene
    # from matplotlib import pyplot as plt
    # N_POINT_THRESH = 25
    # n_insts = []
    # for sid in range(len(dataset)):
    #     pcs, segms, flows, _ = dataset[sid]
    #     segm1 = segms[0]
    #     inst_ids, inst_sizes = np.unique(segm1, return_counts=True)
    #     inst_ids_keep = inst_ids[inst_sizes >= N_POINT_THRESH]
    #     n_insts.append(inst_ids_keep.shape[0])
    # n_insts = np.sort(np.array(n_insts))
    # print (n_insts.min(), n_insts.mean(), n_insts.max())
    # plt.plot(n_insts)
    # plt.savefig('n_inst_waymo_train.png')
    # exit()

    interval = 50
    for sid in range(len(dataset)):
        if downsampled:
            pcs, segms, flows, _ = dataset[sid]
        else:
            pcs, segms, flows = dataset[sid]
        pc1, pc2 = pcs[0], pcs[1]
        segm1, segm2 = segms[0], segms[1]
        flow = flows[0]

        pcds = []
        pcds.append(build_pointcloud(pc1, segm1, with_background=True))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0]))
        pcds.append(build_pointcloud(pc2, segm2, with_background=True).translate([interval, 0.0, 0.0]))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[interval, 0, 0]))

        # # Check moving objects according to GT ego-motion
        # sequence_name, view_id1, view_id2 = dataset.data_ids[sid]
        # sequence_path = osp.join('/home/ziyang/Desktop/Datasets/Waymo/data', sequence_name)
        # pose1, pose2 = np.load(osp.join(sequence_path, 'pose_%04d.npy'%(view_id1))), np.load(osp.join(sequence_path, 'pose_%04d.npy'%(view_id2)))
        # rot1, transl1 = pose1[0:3, 0:3], pose1[0:3, 3]
        # rot2, transl2 = pose2[0:3, 0:3], pose2[0:3, 3]
        # rot = rot2.T @ rot1
        # transl = rot2.T @ (transl1 - transl2)
        # ego_flow = np.einsum('ij,nj->ni', rot, pc1) + transl - pc1
        # diff = np.linalg.norm(flow - ego_flow, axis=1)
        # moving = (diff > 0.02).astype(np.int32)
        # pcds.append(build_pointcloud(pc1, moving, with_background=True).translate([-interval, 0.0, 0.0]))
        # pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[-interval, 0, 0]))

        # Check flows
        pcds.append(build_pointcloud(pc1, np.zeros_like(segm1)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc1 + flow, np.zeros_like(segm1)).translate([-1 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-1 * interval, 0.0, 0.0]))

        o3d.visualization.draw_geometries(pcds)