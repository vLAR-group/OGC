import os
import os.path as osp
import tqdm
import yaml
import argparse
import json

import torch
from torch.utils.data import DataLoader

from losses.seg_loss_unsup import fit_motion_svd_batch, interpolate_mask_by_flow, match_mask_by_iou
from metrics.flow_metric import eval_flow
from utils.pytorch_util import AverageMeter


def weighted_kabsch(pc, flow, mask):
    """
    :param pc: (B, N, 3) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :param mask: (B, N, K) torch.Tensor.
    :return:
        flow: (B, N, 3) torch.Tensor.
    """
    n_batch, n_point, n_object = mask.size()
    mask = mask.transpose(1, 2)
    mask = mask.reshape(n_batch * n_object, n_point)
    pc_rep = pc.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
    flow_rep = flow.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

    # Estimate the rigid transformation
    object_R, object_t = fit_motion_svd_batch(pc_rep, pc_rep + flow_rep, mask)

    # Apply the estimated rigid transformation onto point cloud
    pc_transformed = torch.einsum('bij,bnj->bni', object_R, pc_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
    pc_transformed = pc_transformed.reshape(n_batch, n_object, n_point, 3)
    mask = mask.reshape(n_batch, n_object, n_point)
    flow = torch.einsum('bkn,bkni->bni', mask, pc_transformed) - pc
    return flow


def object_aware_icp(pc1, pc2, flow, mask1, mask2, icp_iter=10, temperature=0.01):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :param mask1: (B, N, K) torch.Tensor.
    :param mask2: (B, N, K) torch.Tensor.
    :return:
        flow_update: (B, N, 3) torch.Tensor.
    """
    # Aligh the object ordering in two frames
    mask2_interpolated = interpolate_mask_by_flow(pc1, pc2, mask1, flow)
    perm = match_mask_by_iou(mask2_interpolated, mask2)
    mask2 = torch.einsum('bij,bnj->bni', perm, mask2)

    # Compute object consistency scores
    consistency12 = torch.einsum('bmk,bnk->bmn', mask1, mask2)

    n_batch, n_point, n_object = mask1.size()
    mask1, mask2 = mask1.transpose(1, 2), mask2.transpose(1, 2)
    mask1_rep = mask1.reshape(n_batch * n_object, n_point)
    pc1_rep = pc1.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

    for iter in range(icp_iter):
        # Compute soft correspondence scores from nearest-neighbor distances
        dist12 = -torch.cdist(pc1 + flow, pc2) / temperature
        corr12 = dist12.softmax(-1)

        # Filter correspondence scores by object consistency scores
        corr12 = corr12 * consistency12
        row_sum = corr12.sum(-1, keepdim=True).clamp(1e-10)
        corr12 = corr12 / row_sum

        # Update scene flow from object-aware soft correspondences
        flow = torch.einsum('bmn,bnj->bmj', corr12, pc2) - pc1

        flow_rep = flow.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
        # Estimate the rigid transformation
        object_R, object_t = fit_motion_svd_batch(pc1_rep, pc1_rep + flow_rep, mask1_rep)
        # Apply the estimated rigid transformation onto point cloud
        pc1_transformed = torch.einsum('bij,bnj->bni', object_R, pc1_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc1_transformed = pc1_transformed.reshape(n_batch, n_object, n_point, 3)
        flow = torch.einsum('bkn,bkni->bni', mask1, pc1_transformed) - pc1
    return flow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--round', type=int, default=0, help='Which round of iterative optimization')
    parser.add_argument('--test_batch_size', type=int, default=48, help='Batch size in testing')
    parser.add_argument('--save', dest='save', default=False, action='store_true', help='Save flow predictions or not')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Configuration for different dataset
    data_root = args.data['root']
    if args.dataset == 'sapien':
        from models.segnet_sapien import MaskFormer3D
        from datasets.dataset_sapien import SapienDataset as TestDataset
        if args.split == 'test':
            data_root = osp.join(data_root, 'mbs-sapien')
        else:
            data_root = osp.join(data_root, 'mbs-shapepart')
        epe_norm_thresh = 0.01
    elif args.dataset == 'ogcdr':
        from models.segnet_ogcdr import MaskFormer3D
        from datasets.dataset_ogcdr import OGCDynamicRoomDataset as TestDataset
        epe_norm_thresh = 0.01
    elif args.dataset == 'kittisf':
        from models.segnet_kitti import MaskFormer3D
        from datasets.dataset_kittisf import KITTISceneFlowDataset as TestDataset
        if args.split == 'val':
            mapping_path = 'data_prepare/kittisf/splits/val.txt'
        else:
            mapping_path = 'data_prepare/kittisf/splits/train.txt'
        epe_norm_thresh = 0.05
    else:
        raise KeyError('Unrecognized dataset!')

    # Setup the segmentation network
    segnet = MaskFormer3D(n_slot=args.segnet['n_slot'],
                          n_point=args.segnet['n_point'],
                          use_xyz=args.segnet['use_xyz'],
                          n_transformer_layer=args.segnet['n_transformer_layer'],
                          transformer_embed_dim=args.segnet['transformer_embed_dim'],
                          transformer_input_pos_enc=args.segnet['transformer_input_pos_enc']).cuda()

    # Load the trained model weights
    weight_path = osp.join(args.save_path + '_R%d'%(args.round), 'best.pth.tar')
    segnet.load_state_dict(torch.load(weight_path)['model_state'])
    segnet.cuda().eval()
    print('Loaded weights from', weight_path)

    # Setup the dataset
    if args.round > 1:
        predflow_path = 'flowstep3d_R%d'%(args.round - 1)
    else:
        predflow_path = 'flowstep3d'
    if args.dataset in ['sapien', 'ogcdr']:
        view_sels = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
        n_frame = len(view_sels)
        test_set = TestDataset(data_root=data_root,
                               split=args.split,
                               view_sels=view_sels,
                               decentralize=args.data['decentralize'])
        test_set_predflow = TestDataset(data_root=data_root,
                                        split=args.split,
                                        view_sels=view_sels,
                                        predflow_path=predflow_path,
                                        decentralize=args.data['decentralize'])
    else:
        n_frame = 2
        test_set = TestDataset(data_root=data_root,
                               mapping_path=mapping_path,
                               downsampled=True,
                               decentralize=args.data['decentralize'])
        test_set_predflow = TestDataset(data_root=data_root,
                                        mapping_path=mapping_path,
                                        downsampled=True,
                                        predflow_path=predflow_path,
                                        decentralize=args.data['decentralize'])
    batch_size = args.test_batch_size

    # Hyperparam for Object-Aware ICP
    icp_iters = {1: 20, 2: 10, 3: 5, 4: 3}
    icp_iter = icp_iters[args.round]

    # Save updated flow predictions
    if args.save:
        assert batch_size % n_frame == 0, \
            'Frame pairs of one scene should be in the same batch, otherwise very inconvenient for saving!'
        # Path to save flow predictions
        SAVE_DIR = osp.join(data_root, 'flow_preds/flowstep3d' + '_R%d'%(args.round))
        os.makedirs(SAVE_DIR, exist_ok=True)
        # Write information about "view_sel" into a meta file
        if args.dataset in ['sapien', 'ogcdr']:
            SAVE_META = SAVE_DIR + '.json'
            with open(SAVE_META, 'w') as f:
                json.dump({'view_sel': view_sels}, f)


    # Iterate over samples
    eval_meter = AverageMeter()
    eval_meter_kabsch = AverageMeter()
    eval_meter_oaicp = AverageMeter()

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader_predflow = DataLoader(test_set_predflow, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    with tqdm.tqdm(enumerate(zip(test_loader, test_loader_predflow), 0), total=len(test_loader), desc='test') as tbar:
        for i, (batch1, batch2) in tbar:
            pcs, _, flows, _ = batch1
            _, _, flow_preds, _ = batch2
            pc1, pc2 = pcs[:, 0].contiguous(), pcs[:, 1].contiguous()
            flow, flow_pred = flows[:, 0].contiguous(), flow_preds[:, 0].contiguous()

            # Forward inference: segmentation
            pc1, pc2, flow_pred = pc1.cuda(), pc2.cuda(), flow_pred.cuda()
            mask1 = segnet(pc1, pc1).detach()
            mask2 = segnet(pc2, pc2).detach()

            # Upadate flow predictions using Weighted Kabsch
            flow_pred_kabsch = weighted_kabsch(pc1, flow_pred, mask1.detach())

            # Upadate flow predictions using OA-ICP
            flow_pred_oaicp = object_aware_icp(pc1, pc2, flow_pred, mask1, mask2, icp_iter=icp_iter)

            # Monitor the change of flow accuracy
            epe, acc_strict, acc_relax, outlier = eval_flow(flow, flow_pred, epe_norm_thresh=epe_norm_thresh)
            eval_meter.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})
            epe_r, acc_strict_r, acc_relax_r, outlier_r = eval_flow(flow, flow_pred_kabsch, epe_norm_thresh=epe_norm_thresh)
            eval_meter_kabsch.append_loss({'EPE': epe_r, 'AccS': acc_strict_r, 'AccR': acc_relax_r, 'Outlier': outlier_r})
            epe_update, acc_strict_update, acc_relax_update, outlier_update = eval_flow(flow, flow_pred_oaicp, epe_norm_thresh=epe_norm_thresh)
            eval_meter_oaicp.append_loss({'EPE': epe_update, 'AccS': acc_strict_update, 'AccR': acc_relax_update, 'Outlier': outlier_update})

            # Save
            if args.save:
                test_set._save_predflow(flow_pred_oaicp, save_root=SAVE_DIR, batch_size=batch_size, n_frame=n_frame, offset=i)

    eval_avg = eval_meter.get_mean_loss_dict()
    print('Original flow:', eval_avg)
    eval_avg_kabsch = eval_meter_kabsch.get_mean_loss_dict()
    print('Weighted Kabsch flow:', eval_avg_kabsch)
    eval_avg_oaicp = eval_meter_oaicp.get_mean_loss_dict()
    print('Object-Aware ICP flow:', eval_avg_oaicp)