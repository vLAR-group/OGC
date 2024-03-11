import os
import os.path as osp
import tqdm
import yaml
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics.seg_metric import accumulate_eval_results, calculate_AP, calculate_PQ_F1, ClusteringMetrics
from utils.pytorch_util import AverageMeter


def pairwise_correspondence(pc1, pc2, flow, temperature=0.01):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :return:
        corr: (B, N, N) torch.Tensor.
    """
    # Compute soft correspondence scores from nearest-neighbor distances
    dist = -torch.cdist(pc1 + flow, pc2) / temperature
    corr = dist.softmax(-1)
    return corr


def collect_correspondences(pc, flows):
    """
    :param pc: (T, N, 3) torch.Tensor.
    :param flows: (T-1, 2, N, 3) torch.Tensor, adjacent pairwise flows.
    :return:
        corrs: a dict containing all pair-wise correpondence.
    """
    n_frame, n_point, _ = pc.size()
    corrs = {}

    # Collect self-correspondence (identity)
    corr = torch.eye(n_point).unsqueeze(0)
    for t in range(n_frame):
        corrs['%d_%d'%(t, t)] = corr

    # Collect adjacent pairwise correspondence from available flow estimations
    for t in range(n_frame - 1):
        corrs['%d_%d'%(t, t+1)] = pairwise_correspondence(pc[t:(t+1)], pc[(t+1):(t+2)], flows[t:(t+1), 0])
        corrs['%d_%d'%(t+1, t)] = pairwise_correspondence(pc[(t+1):(t+2)], pc[t:(t+1)], flows[t:(t+1), 1])
    
    # Collect adjacent pairwise correspondence by propagation
    for interval in range(2, n_frame):
        for t in range(0, n_frame - interval):
            corr = torch.bmm(corrs['%d_%d'%(t, t+interval-1)], corrs['%d_%d'%(t+interval-1, t+interval)])
            corrs['%d_%d'%(t, t+interval)] = corr / corr.sum(-1, keepdim=True).clamp(1e-10)
            corr = torch.bmm(corrs['%d_%d'%(t+interval, t+interval-1)], corrs['%d_%d'%(t+interval-1, t)])
            corrs['%d_%d'%(t+interval, t)] = corr / corr.sum(-1, keepdim=True).clamp(1e-10)

    return corrs


def match_mask_by_cost(mask1, mask2, measure='ce'):
    """
    :param mask1: (N, K) torch.Tensor.
    :param mask2: (N, K) torch.Tensor.
    :return:
        :param mask2: (N, K) torch.Tensor.
    """
    n_object = mask1.shape[-1]

    mask1_rep = mask1.unsqueeze(2).repeat(1, 1, n_object)
    mask2_rep = mask2.unsqueeze(1).repeat(1, n_object, 1)

    # Match objects in two frames with Hungarian to minimize cross-entropy
    if measure == 'ce':
        cost = F.binary_cross_entropy(mask1_rep, mask2_rep, reduction='none')
        cost = cost.mean(0)
        cost = cost.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost, maximize=False)
    # Match objects in two frames with Hungarian to maximize IoU
    else:
        intersection = (mask1_rep * mask2_rep).sum(0)
        union = (mask1_rep + mask2_rep).sum(0)
        iou = intersection / union.clamp(1e-10)
        iou = iou.cpu().numpy()
        _, col_ind = linear_sum_assignment(iou, maximize=True)
    perm = torch.eye(n_object, dtype=torch.float32, device=mask2.device)[col_ind]

    # Reorder objects in 2nd frame
    mask2 = torch.einsum('ij,nj->ni', perm, mask2)
    return mask2


def mask_voting(pc, mask, flows, time_window_size=3):
    """
    :param pc: (T, N, 3) torch.Tensor.
    :param mask: (T, N, K) torch.Tensor.
    :param flows: (T-1, 2, N, 3) torch.Tensor, adjacent pairwise flows.
    :return:
        mask: (T, N, K) torch.Tensor.
    """
    n_frame, n_point, _ = pc.size()
    mask_voted = []

    # Get pair-wise correpondence
    corrs = collect_correspondences(pc, flows)

    for t in range(n_frame):
        votes = []
        mask_t = mask[t]
        time_window = list(range(max(0, t-time_window_size), min(n_frame, t+time_window_size+1)))

        for v in time_window:
            if v == t:
                votes.append(mask_t)
            else:
                # Accumulate votes from other frames
                corr = corrs['%d_%d'%(t, v)][0]
                mask_v = torch.einsum('mn,nk->mk', corr, mask[v])
                # Aligh the object ordering in two frames
                mask_v = match_mask_by_cost(mask_t, mask_v)
                votes.append(mask_v)

        # Aggregate the votes
        vote = torch.stack(votes, 0).mean(0)
        # Normlaize
        vote = vote / vote.sum(-1, keepdim=True).clamp(1e-10)
        mask_voted.append(vote)

    mask_voted = torch.stack(mask_voted, 0)
    return mask_voted
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--round', type=int, default=0, help='Trained segmentation model of which round')
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true', help='Qualitative / Quantitative evaluation mode')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size in testing')
    parser.add_argument('--time_window_size', type=int, default=3, help='Time window for multi-frame co-segmentation')
    parser.add_argument('--use_gt_flow', dest='use_gt_flow', default=False, action='store_true', help='Use GT flows in co-segmentation or not')
    parser.add_argument('--save', dest='save', default=False, action='store_true', help='Save segmentation predictions or not')

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
    elif args.dataset == 'ogcdr':
        from models.segnet_ogcdr import MaskFormer3D
        from datasets.dataset_ogcdr import OGCDynamicRoomDataset as TestDataset
    elif args.dataset == 'kittisf':
        from models.segnet_kitti import MaskFormer3D
        from datasets.dataset_kittisf import KITTISceneFlowDataset as TestDataset
        if args.split == 'val':
            mapping_path = 'data_prepare/kittisf/splits/val.txt'
        else:
            mapping_path = 'data_prepare/kittisf/splits/train.txt'
    elif args.dataset == 'kittidet':
        from models.segnet_kitti import MaskFormer3D
        from datasets.dataset_kittidet import KITTIDetectionDataset as TestDataset
        if args.split == 'val':
            mapping_path = 'data_prepare/kittidet/splits/val.txt'
        else:
            mapping_path = 'data_prepare/kittidet/splits/train.txt'
    else:
        raise KeyError('Unrecognized dataset!')

    # Setup the network
    segnet = MaskFormer3D(n_slot=args.segnet['n_slot'],
                          n_point=args.segnet['n_point'],
                          use_xyz=args.segnet['use_xyz'],
                          n_transformer_layer=args.segnet['n_transformer_layer'],
                          transformer_embed_dim=args.segnet['transformer_embed_dim'],
                          transformer_input_pos_enc=args.segnet['transformer_input_pos_enc']).cuda()

    # Load the trained model weights
    if args.round > 0:
        weight_path = osp.join(args.save_path + '_R%d'%(args.round), 'best.pth.tar')
    else:
        weight_path = osp.join(args.save_path, 'best.pth.tar')
    segnet.load_state_dict(torch.load(weight_path)['model_state'])
    segnet.cuda().eval()
    print('Loaded weights from', weight_path)

    # Setup the scene flow source
    if args.use_gt_flow:
        predflow_path = None
    else:
        if args.round > 1:
            predflow_path = args.predflow_path + '_R%d' % (args.round - 1)
        else:
            predflow_path = args.predflow_path

    # Setup the dataset
    if args.dataset in ['sapien', 'ogcdr']:
        view_sels = [[0, 1], [1, 2], [2, 3], [3, 2]]
        n_frame = len(view_sels)
        test_set = TestDataset(data_root=data_root,
                               split=args.split,
                               view_sels=view_sels,
                               predflow_path=predflow_path,
                               decentralize=args.data['decentralize'])
        ignore_npoint_thresh = 0
    else:
        if args.dataset == 'kittisf':
            view_sels = [[0, 1], [1, 0]]
            n_frame = len(view_sels)
            test_set = TestDataset(data_root=data_root,
                                   mapping_path=mapping_path,
                                   downsampled=True,
                                   view_sels=view_sels,
                                   predflow_path=predflow_path,
                                   decentralize=args.data['decentralize'])
        else:
            n_frame = 1
            test_set = TestDataset(data_root=data_root,
                                   mapping_path=mapping_path,
                                   decentralize=args.data['decentralize'])
        ignore_npoint_thresh = 50
    batch_size = args.test_batch_size


    # Qualitative evaluation mode
    if args.visualize:
        import open3d as o3d
        from utils.visual_util import build_pointcloud

        if args.dataset in ['sapien', 'ogcdr']:
            test_loader = DataLoader(test_set, batch_size=n_frame, shuffle=False, pin_memory=True, num_workers=4)
            h_interval = -1.5
            w_interval = 1.5
            with_background = False
        else:
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
            w_interval = 50
            with_background = True

        with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
            for i, batch in tbar:
                pcs, segms, flows, _ = batch
                pc = pcs[:, 0].contiguous().cuda()
                segm = segms[:, 0].contiguous()     # Groundtruth segmentation
                flows = flows[:(n_frame - 1)].contiguous().cuda()        # Remove redundancy

                # Forward inference
                mask = segnet(pc, pc).detach()

                # Aggregate multi-frame results by voting
                mask_voted = mask_voting(pc, mask, flows, time_window_size=args.time_window_size)

                mask = mask.cpu().numpy()
                segm_pred = mask.argmax(2)
                mask_voted = mask_voted.cpu().numpy()
                segm_pred_voted = mask_voted.argmax(2)

                # Visualize
                pc = pc.detach().cpu().numpy()
                segm = segm.numpy()
                pcds = []
                if args.dataset in ['sapien', 'ogcdr']:
                    for t in range(segm.shape[0]):
                        pcds.append(build_pointcloud(pc[t], segm[t], with_background=with_background).translate([t*w_interval, 0.0, 0.0]))
                        pcds.append(build_pointcloud(pc[t], segm_pred[t], with_background=with_background).translate([t*w_interval, h_interval, 0.0]))
                        pcds.append(build_pointcloud(pc[t], segm_pred_voted[t], with_background=with_background).translate([t*w_interval, 2*h_interval, 0.0]))
                else:
                    pcds.append(build_pointcloud(pc[0], segm[0], with_background=with_background).translate([0.0, 0.0, 0.0]))
                    pcds.append(build_pointcloud(pc[0], segm_pred[0], with_background=with_background).translate([w_interval, 0.0, 0.0]))
                o3d.visualization.draw_geometries(pcds)

    # Quantitative evaluation mode
    else:
        assert batch_size % n_frame == 0, \
            'Frames of one scene should be in the same batch, otherwise very inconvenient for evaluation!'
        # Save segmentation predictions
        if args.save:
            # Path to save segmentation predictions
            SAVE_DIR = osp.join(data_root, 'segm_preds/Vote' + '_T%d'%(args.time_window_size))
            os.makedirs(SAVE_DIR, exist_ok=True)

        # Iterate over the dataset
        mbs_eval = ClusteringMetrics(spec=[ClusteringMetrics.IOU, ClusteringMetrics.RI])
        eval_meter = AverageMeter()
        ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

        with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
            for i, batch in tbar:
                pcs, segms, flows, _ = batch
                pc = pcs[:, 0].contiguous().cuda()
                segm = segms[:, 0].contiguous()     # Groundtruth segmentation

                # Forward inference
                mask = segnet(pc, pc).detach()

                # Aggregate multi-frame results by voting
                mask_voted = []
                for sid in range(segm.shape[0] // n_frame):
                    pc_s = pc[(n_frame * sid):(n_frame * (sid + 1))]
                    mask_s = mask[(n_frame * sid):(n_frame * (sid + 1))]
                    flows_s = flows[(n_frame * sid):(n_frame * (sid + 1) - 1)].contiguous().cuda()
                    mask_voted_s = mask_voting(pc_s, mask_s, flows_s, time_window_size=args.time_window_size)
                    mask_voted.append(mask_voted_s)
                mask_voted = torch.cat(mask_voted, 0)

                # Accumulate for AP, PQ, F1, Pre, Rec
                # Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = accumulate_eval_results(segm, mask, ignore_npoint_thresh=ignore_npoint_thresh)
                Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = accumulate_eval_results(segm, mask_voted, ignore_npoint_thresh=ignore_npoint_thresh)
                ap_eval_meter['Pred_IoU'].append(Pred_IoU)
                ap_eval_meter['Pred_Matched'].append(Pred_Matched)
                ap_eval_meter['Confidence'].append(Confidence)
                ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

                # mIoU & RI metrics
                for sid in range(segm.shape[0] // n_frame):
                    # all_mask = mask[(n_frame * sid):(n_frame * (sid + 1))]
                    all_mask = mask_voted[(n_frame * sid):(n_frame * (sid + 1))]
                    all_segm = segm[(n_frame * sid):(n_frame * (sid + 1))].long()
                    per_scan_mbs = mbs_eval(all_mask, all_segm, ignore_npoint_thresh=ignore_npoint_thresh)
                    eval_meter.append_loss({'per_scan_iou_avg': np.mean(per_scan_mbs['iou']),
                                            'per_scan_iou_std': np.std(per_scan_mbs['iou']),
                                            'per_scan_ri_avg': np.mean(per_scan_mbs['ri']),
                                            'per_scan_ri_std': np.std(per_scan_mbs['ri'])})

                # Save
                if args.save:
                    test_set._save_predsegm(mask_voted, save_root=SAVE_DIR, batch_size=batch_size, n_frame=n_frame, offset=i)

        # Evaluate
        print('Evaluation on %s-%s:'%(args.dataset, args.split))
        Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
        Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
        Confidence = np.concatenate(ap_eval_meter['Confidence'])
        N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
        # AP = calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot=True)
        AP = calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot=False)
        print('AveragePrecision@50:', AP)
        PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
        print('PanopticQuality@50:', PQ, 'F1-score@50:', F1, 'Prec@50:', Pre, 'Recall@50:', Rec)
        eval_avg = eval_meter.get_mean_loss_dict()
        print(eval_avg)