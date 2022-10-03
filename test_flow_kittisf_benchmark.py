import tqdm
import yaml
import argparse
import numpy as np
from collections import OrderedDict

import torch

from utils.pytorch_util import AverageMeter
from metrics.flow_metric import eval_flow
from utils.data_util import upsample_feat


def preproc(pc1, pc2, flow, remove_ground=False, n_sample_point=None):
    """
    Follow the same data preprocessing in FlowStep3D.
    :param pc1: (N, 3).
    :param pc2: (N, 3).
    :param flow: (N, 3).
    :return:
        pc1: (N', 3).
        pc2: (N', 3).
        flow: (N', 3).
    """
    if remove_ground:
        is_ground = np.logical_and(pc1[:, 1] < -1.4, pc2[:, 1] < -1.4)
        not_ground = np.logical_not(is_ground)
        pc1, pc2 = pc1[not_ground], pc2[not_ground]
        flow = flow[not_ground]

    # Random sampling
    if n_sample_point is not None:
        indices = pc1.shape[0]
        try:
            sampled_indices1 = np.random.choice(indices, size=n_sample_point, replace=False, p=None)
            sampled_indices2 = np.random.choice(indices, size=n_sample_point, replace=False, p=None)
        except:
            # replicate some points
            sampled_indices1 = np.random.choice(indices, size=n_sample_point, replace=True, p=None)
            sampled_indices2 = np.random.choice(indices, size=n_sample_point, replace=True, p=None)
        pc1, pc2 = pc1[sampled_indices1], pc2[sampled_indices2]
        flow = flow[sampled_indices1]
    return pc1, pc2, flow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Fix the same random seed as FlowStep3D
    np.random.seed(18)
    torch.manual_seed(18)

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
    view_sels = [[0, 1], [1, 0]]
    mapping_path = 'data_prepare/kittisf/splits/kitti142.txt'
    test_set = TestDataset(data_root=data_root,
                           mapping_path=mapping_path,
                           downsampled=False,
                           view_sels=view_sels)
    epe_norm_thresh = 0.05

    # Setup the dataset to load estimated flow
    predflow_path = 'flowstep3d_for-benchmark_R2'
    test_set_predflow = TestDataset(data_root=data_root + '_downsampled',
                                    mapping_path=mapping_path,
                                    downsampled=True,
                                    view_sels=view_sels,
                                    predflow_path=predflow_path)

    # Iterate over the dataset
    eval_meter_fs3d = AverageMeter()
    eval_meter = AverageMeter()
    n_scenes = len(test_set) // 2
    pbar = tqdm.tqdm(total=n_scenes)
    for sid in range(n_scenes):
        # Only evaluate in forward direction
        pcs_org, _, flows_org, _ = test_set[sid * 2]
        pcs, _, flow_preds, _ = test_set_predflow[sid * 2]
        pc1_org, pc2_org, flow_org = pcs_org[0], pcs_org[1], flows_org[0]
        pc, flow_pred = pcs[0], flow_preds[0]

        # Preprocess the data before FlowStep3D
        pc1_org, pc2_org, flow_org = preproc(pc1_org, pc2_org, flow_org, remove_ground=True, n_sample_point=8192)

        # Forward inference with FlowStep3D
        flow_org = torch.from_numpy(flow_org).unsqueeze(0)
        pc1_org = torch.from_numpy(pc1_org).unsqueeze(0).cuda()
        pc2_org = torch.from_numpy(pc2_org).unsqueeze(0).cuda()
        flow_preds_fs3d = flownet(pc1_org, pc2_org, pc1_org, pc2_org, iters=5)
        flow_pred_fs3d = flow_preds_fs3d[-1]

        # Evaluate scene flow estimation from FlowStep3D
        epe, acc_strict, acc_relax, outlier = eval_flow(flow_org, flow_pred_fs3d)
        eval_meter_fs3d.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})

        # Remove the ground from our downsampled version
        is_ground = (pc[:, 1] < -1.4)
        not_ground = np.logical_not(is_ground)
        pc, flow_pred = pc[not_ground], flow_pred[not_ground]
        pc = torch.from_numpy(pc).unsqueeze(0).cuda()
        flow_pred = torch.from_numpy(flow_pred).unsqueeze(0).cuda()
        # Interpolate from our estimation to obtain flow on specified points
        flow_pred = upsample_feat(pc1_org, pc, flow_pred)

        # Evaluate scene flow estimation interpolated from ours
        epe, acc_strict, acc_relax, outlier = eval_flow(flow_org, flow_pred)
        eval_meter.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})

        pbar.update()

    eval_avg_fs3d = eval_meter_fs3d.get_mean_loss_dict()
    print ('FlowStep3D:', eval_avg_fs3d)
    eval_avg = eval_meter.get_mean_loss_dict()
    print ('Ours:', eval_avg)