import os
import os.path as osp
import tqdm
import yaml
import argparse
import numpy as np
from collections import OrderedDict

import torch

from metrics.flow_metric import eval_flow
from utils.pytorch_util import AverageMeter
from utils.data_util import fps_downsample, upsample_feat
from utils.icp_util import icp
from utils.gpf_util import ground_plane_fitting


def register_bound(pc1, pc2, rot, transl, bound=True):
    """
    Register points of frame1 into frame2. Remove registered frame1 points that overpass the frame2 points' bound.
    :param pc1: (N, 3).
    :param pc2: (N, 3).
    :param rot: (3, 3).
    :param transl: (3,).
    :return:
        select_idx: (N,).
    """
    pc1_transformed = np.einsum('ij,nj->ni', rot, pc1) + transl
    select_idx = np.ones(pc1.shape[0], dtype=bool)

    if bound:
        # Bound by FOV: Similar standard to data pre-processing (Note the permutation of x/y/z axis!)
        pc_x90_idx = (pc1_transformed[:, 2] > abs(pc1_transformed[:, 0]))  # front-view only
        not_outrange = ((np.square(pc1_transformed[:, 0]) + np.square(pc1_transformed[:, 1]) + np.square(pc1_transformed[:, 2])) < 60 * 60)  # remove outrange
        not_outbound = (abs(pc1_transformed[:, 0]) < 50)
        within_depth = (pc1_transformed[:, 2] < 35)
        select_idx_fov = pc_x90_idx * not_outrange * not_outbound * within_depth
        select_idx = np.logical_and(select_idx, select_idx_fov)

        # # Bound by bounding box (No major changes to FOV bound results)
        # pc2_range_min, pc2_range_max = pc2.min(axis=0), pc2.max(axis=0)
        # select_idx_box_min = np.all(pc1_transformed >= pc2_range_min, axis=1)
        # select_idx_box_max = np.all(pc1_transformed <= pc2_range_max, axis=1)
        # select_idx_box = np.logical_and(select_idx_box_min, select_idx_box_max)
        # select_idx = np.logical_and(select_idx, select_idx_box)
    return select_idx


if __name__ == '__main__':
    # import open3d as o3d
    # from utils.visual_util import build_pointcloud, pc_segm_to_sphere
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--use_odometry', dest='use_odometry', default=False, action='store_true',
                        help='Use groundtruth odometry or not')
    parser.add_argument('--denoise', dest='denoise', default=False, action='store_true',
                        help='Denoise abnormally large-scale flow predictions or not')
    parser.add_argument('--bound', dest='bound', default=False, action='store_true',
                        help='Perform register bounding or not')
    parser.add_argument('--test_model_iters', type=int, default=4,
                        help='Number of FlowStep3D model unrolling iterations in testing')
    parser.add_argument('--save', dest='save', default=False, action='store_true', help='Save flow predictions or not')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Setup the network
    from models.flownet_kitti import FlowStep3D
    flownet = FlowStep3D(npoint=args.flownet['npoint'],
                         use_instance_norm=args.flownet['use_instance_norm'],
                         loc_flow_nn=args.flownet['loc_flow_nn'],
                         loc_flow_rad=args.flownet['loc_flow_rad'],
                         k_decay_fact=0.5).cuda()

    # Load the trained model weights
    weight_path = args.save_path
    weights_loaded = torch.load(weight_path)['state_dict']
    weights = OrderedDict([(k[6:], v) for k, v in weights_loaded.items()])
    flownet.load_state_dict(weights)
    flownet.eval()
    print('Loaded weights from', weight_path)

    # Setup the dataset
    from datasets.dataset_waymo import WaymoOpenDataset as TestDataset
    data_root = args.data['root']
    if args.split == 'val':
        mapping_path = 'data_prepare/waymo/splits/val.txt'
    else:
        mapping_path = 'data_prepare/waymo/splits/train.txt'
    test_set = TestDataset(data_root=data_root,
                           mapping_path=mapping_path)
    epe_norm_thresh = 0.05

    # Save flow predictions
    if args.save:
        # Path to save flow predictions
        SAVE_DIR = osp.join(data_root, 'flow_preds/flowstep3d_gpf')
        if args.use_odometry:
            SAVE_DIR += '_odo'
        if args.bound:
            SAVE_DIR += '_bound'
        if args.denoise:
            SAVE_DIR += '_denoise'
        os.makedirs(SAVE_DIR, exist_ok=True)

    # Hyperparameters for ground plane removal
    n_point_gpf = 2048
    n_gpf_iter = 5
    n_gpf_lpr = 50
    thresh_seed = 0.4
    thresh_dist = 0.4

    # Hyperparameters for ICP
    n_point_icp = 1024
    max_icp_iters = 50
    decentralize = True

    # Threshold to remove outliers in flow predictions
    thresh_flow_norm = 2.5

    # Iterate over the dataset
    eval_meter = AverageMeter()
    eval_meter_g = AverageMeter()
    eval_meter_ng = AverageMeter()
    n_scenes = len(test_set)
    pbar = tqdm.tqdm(total=n_scenes)

    sids = list(range(n_scenes))
    # np.random.shuffle(sids)


    for sid in sids:
        sequence_name, view_id1, view_id2 = test_set.data_ids[sid]
        print (sequence_name, view_id1, view_id2)

        pcs, _, flows = test_set[sid]
        pc1_org, pc2_org = pcs[0], pcs[1]
        flow_org = flows[0]      # Groundtruth flow

        # Occasionally, there are empty frames
        if min(pc1_org.shape[0], pc2_org.shape[0]) < 1:
            flow_pred_org = np.zeros_like(pc1_org)
            flow_pred_org = torch.from_numpy(flow_pred_org).unsqueeze(0)
            if args.save:
                test_set._save_predflow(flow_pred_org, save_root=SAVE_DIR, batch_size=1, n_frame=1, offset=sid)
            pbar.update()
            continue

        # Extract points above the ground: height threshold + GPF (ground plane fitting)
        is_ground1_heighthresh = (pc1_org[:, 1] < 0.3)
        is_ground1_gpf = ground_plane_fitting(pc1_org,
                                              n_sample_point=n_point_gpf,
                                              n_iter=n_gpf_iter,
                                              n_lpr=n_gpf_lpr,
                                              thresh_seed=thresh_seed,
                                              thresh_dist=thresh_dist).astype(bool)
        is_ground1 = np.logical_or(is_ground1_heighthresh, is_ground1_gpf)
        not_ground1 = np.logical_not(is_ground1)
        is_ground2_heighthresh = (pc2_org[:, 1] < 0.3)
        is_ground2_gpf = ground_plane_fitting(pc2_org,
                                              n_sample_point=n_point_gpf,
                                              n_iter=n_gpf_iter,
                                              n_lpr=n_gpf_lpr,
                                              thresh_seed=thresh_seed,
                                              thresh_dist=thresh_dist).astype(bool)
        is_ground2 = np.logical_or(is_ground2_heighthresh, is_ground2_gpf)
        not_ground2 = np.logical_not(is_ground2)
        pc1, pc2 = pc1_org[not_ground1], pc2_org[not_ground2]
        flow = flow_org[not_ground1]

        # # Check the ground plane removal by visualization
        # pcds = []
        # interval = 60
        # pcds.append(build_pointcloud(pc1_org, np.zeros(pc1_org.shape[0], dtype=np.int32), with_background=True))
        # pcds.append(build_pointcloud(pc2_org, np.ones(pc2_org.shape[0], dtype=np.int32), with_background=True))
        # pcds.append(build_pointcloud(pc1, np.zeros(pc1.shape[0], dtype=np.int32), with_background=True).translate([interval, 0, 0]))
        # pcds.append(build_pointcloud(pc2, np.ones(pc2.shape[0], dtype=np.int32), with_background=True).translate([interval, 0, 0]))
        # o3d.visualization.draw_geometries(pcds)
        # continue

        # Use groundtruth ego-motion
        if args.use_odometry:
            sequence_path = osp.join(data_root, 'data', sequence_name)
            pose1, pose2 = np.load(osp.join(sequence_path, 'pose_%04d.npy' % (view_id1))), np.load(
                osp.join(sequence_path, 'pose_%04d.npy' % (view_id2)))
            rot1, transl1 = pose1[0:3, 0:3], pose1[0:3, 3]
            rot2, transl2 = pose2[0:3, 0:3], pose2[0:3, 3]
            rot = rot2.T @ rot1
            transl = rot2.T @ (transl1 - transl2)

        # Use ICP to fit ego-motion
        else:
            # Decentralize point clouds for ICP
            if decentralize:
                center = np.concatenate((pc1, pc2), 0).mean(0)
                pc1_icp = pc1 - center
                pc2_icp = pc2 - center
            else:
                pc1_icp, pc2_icp = pc1, pc2

            # Downsample points before ICP
            fps_idx1 = fps_downsample(pc1_icp, n_sample_point=n_point_icp)
            pc1_fps = pc1_icp[fps_idx1]
            fps_idx2 = fps_downsample(pc2_icp, n_sample_point=n_point_icp)
            pc2_fps = pc2_icp[fps_idx2]

            # Fit transformation of background points (camera ego-motion) with ICP
            T, _, _ = icp(pc1_fps, pc2_fps, max_iterations=max_icp_iters)
            rot, transl = T[:3, :3], T[:3, 3]

        # Regard fitted transformation as flows
        flow_pred_org_icp = np.einsum('ij,nj->ni', rot, pc1_org) + transl - pc1_org
        flow_pred_org_icp = flow_pred_org_icp.astype(np.float32)
        flow_pred_org = flow_pred_org_icp.copy()
        flow_pred_org = torch.from_numpy(flow_pred_org).unsqueeze(0)

        # Apply fitted transformation to (above ground) points in frame 1
        pc1 = np.einsum('ij,nj->ni', rot, pc1) + transl
        pc1 = pc1.astype(np.float32)

        # Bound and remove points with no correspondence in the other frame
        select_idx1 = register_bound(pc1, pc2, rot, transl, args.bound)
        pc1_sel = pc1[select_idx1]
        inv_rot, inv_transl = rot.T, np.einsum('ij,j->i', -rot.T, transl)
        select_idx2 = register_bound(pc2, pc1_sel, inv_rot, inv_transl, args.bound)
        pc2_sel = pc2[select_idx2]

        # # Check if bounding works
        # pcds = []
        # point_radius = 0.05
        # resolution = 3
        # pc1_vis = pc_segm_to_sphere(pc1_sel, np.zeros(pc1_sel.shape[0], dtype=np.int32),
        #                                 radius=point_radius, resolution=resolution)
        # pc2_vis = pc_segm_to_sphere(pc2_sel, np.ones(pc2_sel.shape[0], dtype=np.int32), radius=point_radius,
        #                             resolution=resolution)
        # pcds.append((pc1_vis + pc2_vis).translate([2 * interval, 0, 0]))
        # o3d.visualization.draw_geometries(pcds)
        # continue

        # Occasionally, no points left after ground removal and bounding
        if min(pc1_sel.shape[0], pc2_sel.shape[0]) > 0:
            # Downsample points before the scene flow network
            fps_idx1 = fps_downsample(pc1_sel, n_sample_point=args.flownet['npoint'])
            pc1_fps = torch.from_numpy(pc1_sel[fps_idx1]).unsqueeze(0).cuda()
            fps_idx2 = fps_downsample(pc2_sel, n_sample_point=args.flownet['npoint'])
            pc2_fps = torch.from_numpy(pc2_sel[fps_idx2]).unsqueeze(0).cuda()

            # Forward inference
            flow_preds = flownet(pc1_fps, pc2_fps, pc1_fps, pc2_fps, iters=args.test_model_iters)
            flow_pred_fps = flow_preds[-1].detach()

            # Upsample flow predictions to original (above ground & bounded) point cloud
            pc1_sel = torch.from_numpy(pc1_sel).unsqueeze(0).cuda()
            flow_pred_sel = upsample_feat(pc1_sel, pc1_fps, flow_pred_fps)
            flow_pred_sel = flow_pred_sel.cpu()

            # Remove outliers in flow predictions
            if args.denoise:
                flow_pred_norm = flow_pred_sel.norm(dim=2)
                outlier_norm = (flow_pred_norm > thresh_flow_norm)
                flow_pred_sel[outlier_norm] = 0

            # Merge flow predictions of the whole scene
            flow_pred = torch.zeros((1, pc1.shape[0], 3), dtype=torch.float32)
            flow_pred[:, select_idx1] = flow_pred_sel
            flow_pred_org[:, not_ground1] += flow_pred

        # # Visualize the scene flow error
        # flow_pred_org = flow_pred_org[0].numpy()

        # flow_scale_min = min(flow_org.min(), flow_pred_org_icp.min(), flow_pred_org.min())
        # flow_scale_max = max(flow_org.max(), flow_pred_org_icp.max(), flow_pred_org.max())
        # pcds = []
        # interval = 60
        # pcds.append(build_pointcloud_flow(pc1_org, flow_org, (flow_scale_min, flow_scale_max)))
        # pcds.append(build_pointcloud_flow(pc1_org, flow_pred_org_icp, (flow_scale_min, flow_scale_max)).translate([interval, 0, 0]))
        # pcds.append(build_pointcloud_flow(pc1_org, flow_pred_org, (flow_scale_min, flow_scale_max)).translate([2 * interval, 0, 0]))
        # o3d.visualization.draw_geometries(pcds)

        # meshes = []
        # interval = 60
        # meshes.append(pc_flow_to_sphere(pc1_org, flow_org, radius=0.02, resolution=5))
        # meshes.append(pc_flow_to_sphere(pc1_org, flow_pred_org_icp, radius=0.02, resolution=5).translate([interval, 0, 0]))
        # meshes.append(pc_flow_to_sphere(pc1_org, flow_pred_org, radius=0.02, resolution=5).translate([2 * interval, 0, 0]))
        # o3d.visualization.draw_geometries(meshes)
        # continue

        # Evaluate
        flow_org = torch.from_numpy(flow_org).unsqueeze(0)
        epe, acc_strict, acc_relax, outlier = eval_flow(flow_org, flow_pred_org, epe_norm_thresh=epe_norm_thresh)
        eval_meter.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})

        is_ground1 = np.logical_not(not_ground1)
        epe_g, acc_strict_g, acc_relax_g, outlier_g = eval_flow(flow_org[:, is_ground1], flow_pred_org[:, is_ground1], epe_norm_thresh=epe_norm_thresh)
        eval_meter_g.append_loss({'EPE': epe_g, 'AccS': acc_strict_g, 'AccR': acc_relax_g, 'Outlier': outlier_g})
        epe_ng, acc_strict_ng, acc_relax_ng, outlier_ng = eval_flow(flow_org[:, not_ground1], flow_pred_org[:, not_ground1], epe_norm_thresh=epe_norm_thresh)
        eval_meter_ng.append_loss({'EPE': epe_ng, 'AccS': acc_strict_ng, 'AccR': acc_relax_ng, 'Outlier': outlier_ng})

        # Save
        if args.save:
            test_set._save_predflow(flow_pred_org, save_root=SAVE_DIR, batch_size=1, n_frame=1, offset=sid)

        pbar.update()

    # Accumulate evaluation results
    eval_avg = eval_meter.get_mean_loss_dict()
    print('Evaluation on waymo-%s:'%(args.split), eval_avg)
    eval_avg_g = eval_meter_g.get_mean_loss_dict()
    print('Ground points:', eval_avg_g)
    eval_avg_ng = eval_meter_ng.get_mean_loss_dict()
    print('Above ground points:', eval_avg_ng)