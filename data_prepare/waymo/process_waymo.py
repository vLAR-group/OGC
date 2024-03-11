import os
import os.path as osp
import pathlib
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def process_flow(flow, pc2, pose1, pose2):
    """
    :param flow: (N, 3).
    :param pc2: (N, 3).
    :param pose1: (4, 4).
    :param pose2: (4, 4).
    :return:
        flow: (N, 3).
    """
    # velocity to flow
    flow = flow * 0.1
    # add ego-motion for scene flow
    rot1, transl1 = pose1[0:3, 0:3], pose1[0:3, 3]
    rot2, transl2 = pose2[0:3, 0:3], pose2[0:3, 3]
    inv_rot2 = np.linalg.inv(rot2)
    flow = pc2 - ((pc2 - flow) @ inv_rot2 + transl2 - transl1) @ rot1
    return flow


def box_to_segm(points, boxes, object_ids, class_ids, relax=0.01):
    """
    :param points: (N, 3).
    :param boxes: (K, 7).
    :param object_ids: (K,).
    :param class_ids: (K,).
    :return:
        segm: (N,).
        semantic_segm: (N,).
    """
    n_point = points.shape[0]
    pc = np.copy(points)[:, :3]

    segm = np.zeros(n_point, dtype=np.int32)
    semantic_segm = np.zeros(n_point, dtype=np.int32)
    for k in range(boxes.shape[0]):
        box = boxes[k]
        center = box[0:3]
        lwh = box[3:6]
        l, w, h = lwh[0], lwh[1], lwh[2]
        angle = np.array([-box[6], 0, 0])

        rot = R.from_euler('zyx', angle, degrees=False).as_matrix()

        pc_tr = pc - center
        pc_tr = np.einsum('ij,nj->ni', rot, pc_tr)

        # Select points within bounding box
        within_box_x = np.logical_and(pc_tr[:, 0] > (-l / 2 - relax), pc_tr[:, 0] < (l / 2 + relax))
        within_box_y = np.logical_and(pc_tr[:, 1] > (-h / 2 - relax), pc_tr[:, 1] < (h / 2 + relax))
        within_box_z = np.logical_and(pc_tr[:, 2] > (-w / 2 - relax), pc_tr[:, 2] < (w / 2 + relax))
        within_box = np.logical_and(np.logical_and(within_box_x, within_box_y), within_box_z)

        # Grant segmentation ID to points
        segm[within_box] = object_ids[k]
        semantic_segm[within_box] = class_ids[k]
    return segm, semantic_segm


class WaymoDataset:
    def __init__(self,
                 dataset_cfg,
                 root_path,
                 split='train',
                 class_names=[]):
        self.dataset_cfg = dataset_cfg
        self.data_path = osp.join(root_path, self.dataset_cfg.PROCESSED_DATA_TAG)
        self.flow_path = osp.join(root_path, self.dataset_cfg.SCENE_FLOW_TAG)
        self.split = split
        split_file = osp.join('data_prepare/waymo/splits', self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_file).readlines()]

        self.class_names = class_names
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)


    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = osp.join(self.data_path, sequence_name, '%04d.npy'%(sample_idx))
        points = np.load(lidar_file)  # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
        return points


    def get_flow(self, sequence_name, sample_idx):
        flow_file = osp.join(self.flow_path, sequence_name, '%04d.npy'%(sample_idx))
        scene_flows = np.load(flow_file)
        return scene_flows


    def process_sequence(self, infos, save_path):
        os.makedirs(save_path, exist_ok=True)
        tracking_to_idx, next_obj_id = {}, 1       # Follow tracking annotations to keep segment_ids consistent in a sequence

        # Permutation matrix to adjust the axis order
        perm = np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]], dtype=np.float32)

        for t, info in enumerate(infos):
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            # Load point cloud
            points = self.get_lidar(sequence_name, sample_idx)
            pc, NLZ_flag = points[:, :3], points[:, 5]

            # Process the point cloud
            pc_labeled = (NLZ_flag == -1)
            pc_x90_idx = (pc[:, 0] > abs(pc[:, 1]))  # front-view only
            not_outrange = ((np.square(pc[:, 0]) + np.square(pc[:, 1]) + np.square(pc[:, 2])) < 60 * 60)  # remove outrange
            not_outbound = (abs(pc[:, 1]) < 50)
            within_depth = (pc[:, 0] < 35)
            select_idx = pc_labeled * pc_x90_idx * not_outrange * not_outbound * within_depth   # selected index
            pc = pc[select_idx]

            # Load scene flow
            if t > 0:
                scene_flows = self.get_flow(sequence_name, sample_idx)
                flow = scene_flows[:, :3]
                flow = flow[select_idx]
                pose = info['pose']
                flow = - process_flow(flow, pc, prev_pose, pose)
                prev_pose = np.copy(pose)
            else:
                prev_pose = np.copy(info['pose'])
                flow = None

            # Load bouding box annotations
            annos = info['annos']
            annos = drop_info_with_name(annos, name='unknown')
            gt_boxes_lidar = annos['gt_boxes_lidar']
            gt_classes = annos['name']
            trackings = annos['obj_ids']
            n_point_in_gt = annos['num_points_in_gt']
            if self.dataset_cfg.get('FILTER_EMPTY_BOXES', False):
                mask = (n_point_in_gt > 0)      # Filter empty boxes
                gt_boxes_lidar, gt_classes, trackings = gt_boxes_lidar[mask], gt_classes[mask], trackings[mask]
                selected = keep_arrays_by_name(gt_classes, self.class_names)     # Ignore DON'T CARE classes
                gt_boxes_lidar, gt_classes, trackings = gt_boxes_lidar[selected], gt_classes[selected], trackings[selected]
            
            # Assign integer IDs to tracked objects
            for tracking in trackings:
                if tracking not in tracking_to_idx.keys():
                    tracking_to_idx[tracking] = next_obj_id
                    next_obj_id += 1

            object_ids = np.array([tracking_to_idx[tracking] for tracking in trackings], dtype=np.int32)
            class_ids = np.array([self.class_names.index(class_name)+1 for class_name in gt_classes], dtype=np.int32)
            # Generate point cloud segmentation
            segm, semantic_segm = box_to_segm(pc, gt_boxes_lidar, object_ids, class_ids)

            # Adjust the axis order
            pc = np.einsum('ij,nj->ni', perm, pc)
            if flow is not None:
                flow = np.einsum('ij,nj->ni', perm, flow)

            # import open3d as o3d
            # from utils.visual_util import build_pointcloud
            # pcds = []
            # interval = 60
            # pcds.append(build_pointcloud(pc, segm, with_background=True))
            # pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0]))
            # if flow is not None:
            #     pcds.append(build_pointcloud(pc, np.zeros_like(segm)).translate([interval, 0, 0]))
            #     pcds.append(build_pointcloud(pc + flow, np.ones_like(segm)).translate([interval, 0, 0]))
            # o3d.visualization.draw_geometries(pcds)

            pose_t = info['pose']
            rot, transl = pose_t[0:3, 0:3], pose_t[0:3, 3]
            rot, transl = perm @ rot @ perm.T, perm @ transl
            pose_t[0:3, 0:3] = rot
            pose_t[0:3, 3] = transl
            np.save(osp.join(save_path, 'pose_%04d.npy'%(sample_idx)), pose_t)

            np.save(osp.join(save_path, 'pc_%04d.npy'%(sample_idx)), pc)
            np.save(osp.join(save_path, 'segm_%04d.npy'%(sample_idx)), segm)
            np.save(osp.join(save_path, 'semantic_segm_%04d.npy'%(sample_idx)), semantic_segm)
            if flow is not None:
                np.save(osp.join(save_path, 'flow_%04d_%04d.npy' % (sample_idx, sample_idx-1)), flow)


    def process_all_sequences(self, save_root=None):
        save_root = osp.join(save_root, 'data')
        print('Processing Waymo dataset... Save into %s'%(save_root))
        os.makedirs(save_root, exist_ok=True)
        num_skipped = 0

        for k in range(len(self.sample_sequence_list)):
            sequence_name = osp.splitext(self.sample_sequence_list[k])[0]
            info_path = osp.join(self.data_path, sequence_name, sequence_name+'.pkl')
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not osp.exists(info_path):
                num_skipped += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)

            print("---%d/%d %s"%(k, len(self.sample_sequence_list), sequence_name))
            save_path = osp.join(save_root, sequence_name)
            self.process_sequence(infos, save_path)

        print('Processing finished. Total skipped (unavailable) sequences %d'%(num_skipped))


    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not osp.exists(sequence_file):
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not osp.exists(sequence_file):
                    temp_sequence_file = pathlib.Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if osp.exists(temp_sequence_file):
                        found_sequence_file = temp_sequence_file
                        break
            if not osp.exists(found_sequence_file):
                found_sequence_file = pathlib.Path(str(sequence_file).replace('_with_camera_labels', ''))
            if osp.exists(found_sequence_file):
                sequence_file = found_sequence_file
        return sequence_file


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_root', type=str, help='Root path for the dataset')
    parser.add_argument('--save_root', type=str, help='Save path for the processed dataset')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process (train/val)')
    parser.add_argument('--cfg_file', type=str, default='data_prepare/waymo/waymo_dataset.yaml', help='Specify the config of dataset')
    args = parser.parse_args()

    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)

    dataset = WaymoDataset(dataset_cfg=dataset_cfg,
                           root_path=args.data_root,
                           split=args.split,
                           class_names=['Vehicle', 'Pedestrian', 'Cyclist'])
    dataset.process_all_sequences(save_root=args.save_root)