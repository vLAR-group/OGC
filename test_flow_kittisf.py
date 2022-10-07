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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
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
    from datasets.dataset_kittisf import KITTISceneFlowDataset as TestDataset
    data_root = args.data['root']
    if args.split == 'val':
        mapping_path = 'data_prepare/kittisf/splits/val.txt'
    else:
        mapping_path = 'data_prepare/kittisf/splits/train.txt'
    view_sels = [[0, 1], [1, 0]]
    test_set = TestDataset(data_root=data_root,
                           mapping_path=mapping_path,
                           downsampled=False,
                           view_sels=view_sels)
    epe_norm_thresh = 0.05

    # Save flow predictions
    if args.save:
        # Path to save flow predictions
        SAVE_DIR = osp.join(data_root, 'flow_preds/flowstep3d')
        os.makedirs(SAVE_DIR, exist_ok=True)


    # Hyperparameters for ICP
    n_point_icp = 1024
    max_icp_iters = 50
    decentralize = True

    # Iterate over the dataset
    eval_meter = AverageMeter()
    n_scenes = len(test_set)
    pbar = tqdm.tqdm(total=n_scenes)
    for sid in range(n_scenes):
        pcs, _, flows, _ = test_set[sid]
        pc1_org, pc2_org = pcs[0], pcs[1]
        flow_org = flows[0]      # Groundtruth flow

        # Extract points above the ground
        is_ground = np.logical_and(pc1_org[:, 1] < -1.4, pc2_org[:, 1] < -1.4)
        not_ground = np.logical_not(is_ground)
        pc1, pc2 = pc1_org[not_ground], pc2_org[not_ground]

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
        rot, transl = T[:3, :3], T[:3, 3].transpose()
        # Regard fitted transformation as flows
        flow_pred_org = np.einsum('ij,nj->ni', rot, pc1_org) + transl - pc1_org
        flow_pred_org = flow_pred_org.astype(np.float32)
        flow_pred_org = torch.from_numpy(flow_pred_org).unsqueeze(0)

        # Apply fitted transformation to (above ground) points in frame 1
        pc1 = np.einsum('ij,nj->ni', rot, pc1) + transl
        pc1 = pc1.astype(np.float32)
        # Downsample points before the scene flow network
        fps_idx1 = fps_downsample(pc1, n_sample_point=args.flownet['npoint'])
        pc1_fps = torch.from_numpy(pc1[fps_idx1]).unsqueeze(0).cuda()
        fps_idx2 = fps_downsample(pc2, n_sample_point=args.flownet['npoint'])
        pc2_fps = torch.from_numpy(pc2[fps_idx2]).unsqueeze(0).cuda()

        # Forward inference
        flow_preds = flownet(pc1_fps, pc2_fps, pc1_fps, pc2_fps, iters=args.test_model_iters)
        flow_pred_fps = flow_preds[-1].detach()

        # Upsample flow predictions to original (above ground) point cloud
        pc1 = torch.from_numpy(pc1).unsqueeze(0).cuda()
        flow_pred = upsample_feat(pc1, pc1_fps, flow_pred_fps)
        # Merge flow predictions of the whole scene
        flow_pred_org[:, not_ground] += flow_pred.cpu()

        # Evaluate
        flow_org = torch.from_numpy(flow_org).unsqueeze(0)
        epe, acc_strict, acc_relax, outlier = eval_flow(flow_org, flow_pred_org, epe_norm_thresh=epe_norm_thresh)
        eval_meter.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})

        # Save
        if args.save:
            test_set._save_predflow(flow_pred_org, save_root=SAVE_DIR, batch_size=1, n_frame=len(view_sels), offset=sid)

        pbar.update()

    # Accumulate evaluation results
    eval_avg = eval_meter.get_mean_loss_dict()
    print('Evaluation on kittisf-%s:'%(args.split), eval_avg)