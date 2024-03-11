import os
import os.path as osp
import tqdm
import yaml
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from metrics.seg_metric_ignmask import accumulate_eval_results, calculate_AP, calculate_PQ_F1, ClusteringMetrics
from utils.pytorch_util import AverageMeter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--round', type=int, default=0, help='Trained segmentation model of which round')
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true', help='Qualitative / Quantitative evaluation mode')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size in testing')
    parser.add_argument('--save', dest='save', default=False, action='store_true', help='Save segmentation predictions or not')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Configuration for different dataset
    data_root = args.data['root']
    from models.segnet_kitti import MaskFormer3D
    from datasets.dataset_waymo_singleframe import WaymoOpenDataset as TestDataset
    if args.split == 'val':
        mapping_path = 'data_prepare/waymo/splits/val.txt'
    else:
        mapping_path = 'data_prepare/waymo/splits/train.txt'

    # Setup the network
    segnet = MaskFormer3D(n_slot=args.segnet['n_slot'],
                          n_point=args.segnet['n_point'],
                          use_xyz=args.segnet['use_xyz'],
                          n_transformer_layer=args.segnet['n_transformer_layer'],
                          transformer_embed_dim=args.segnet['transformer_embed_dim'],
                          transformer_input_pos_enc=args.segnet['transformer_input_pos_enc']).cuda()

    # Load the trained model weights
    # if args.round > 0:
    #     weight_path = osp.join(args.save_path + '_R%d'%(args.round), 'current.pth.tar')
    # else:
    #     weight_path = osp.join(args.save_path, 'current.pth.tar')
    if args.round > 0:
        weight_path = osp.join(args.save_path + '_R%d'%(args.round), 'best.pth.tar')
    else:
        weight_path = osp.join(args.save_path, 'best.pth.tar')
    segnet.load_state_dict(torch.load(weight_path)['model_state'])
    segnet.cuda().eval()
    print('Loaded weights from', weight_path)

    # Setup the dataset
    n_frame = 1
    ignore_class_ids = [2, 3]
    ignore_npoint_thresh = 50
    test_set = TestDataset(data_root=data_root,
                           mapping_path=mapping_path,
                           downsampled=True,
                           sampled_interval=1,
                           decentralize=args.data['decentralize'],
                           ignore_class_ids=ignore_class_ids,
                           ignore_npoint_thresh=ignore_npoint_thresh)
    batch_size = args.test_batch_size


    # Qualitative evaluation mode
    if args.visualize:
        import open3d as o3d
        from utils.visual_util import build_pointcloud

        test_loader = DataLoader(test_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
        # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
        w_interval = 60
        with_background = True

        with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
            for i, batch in tbar:
                pcs, segms, _ = batch
                pc = pcs[:, 0].contiguous().cuda()
                segm = segms[:, 0].contiguous()     # Groundtruth segmentation

                # Forward inference
                mask = segnet(pc, pc)
                mask = mask.detach().cpu().numpy()
                segm_pred = mask.argmax(2)

                # Visualize
                pc = pc.detach().cpu().numpy()
                segm = segm.numpy()
                pcds = []
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
            SAVE_DIR = osp.join(data_root, 'segm_preds/OGC' + '_R%d'%(args.round))
            os.makedirs(SAVE_DIR, exist_ok=True)

        # Iterate over the dataset
        mbs_eval = ClusteringMetrics(spec=[ClusteringMetrics.IOU, ClusteringMetrics.RI])
        eval_meter = AverageMeter()
        ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

        with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
            for i, batch in tbar:
                pcs, segms, valids = batch
                ignores = 1 - valids
                pc = pcs[:, 0].contiguous().cuda()
                segm = segms[:, 0].contiguous()     # Groundtruth segmentation
                ignore = ignores[:, 0].contiguous()

                # Forward inference
                mask = segnet(pc, pc)

                # Accumulate for AP, PQ, F1, Pre, Rec
                Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = accumulate_eval_results(segm, mask, ignore)
                ap_eval_meter['Pred_IoU'].append(Pred_IoU)
                ap_eval_meter['Pred_Matched'].append(Pred_Matched)
                ap_eval_meter['Confidence'].append(Confidence)
                ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

                # mIoU & RI metrics
                for sid in range(segm.shape[0] // n_frame):
                    all_mask = mask[(n_frame * sid):(n_frame * (sid + 1))]
                    all_segm = segm[(n_frame * sid):(n_frame * (sid + 1))].long()
                    all_ignore = ignore[(n_frame * sid):(n_frame * (sid + 1))]
                    per_scan_mbs = mbs_eval(all_mask, all_segm, all_ignore)
                    eval_meter.append_loss({'per_scan_iou_avg': np.mean(per_scan_mbs['iou']),
                                            'per_scan_iou_std': np.std(per_scan_mbs['iou']),
                                            'per_scan_ri_avg': np.mean(per_scan_mbs['ri']),
                                            'per_scan_ri_std': np.std(per_scan_mbs['ri'])})

                # Save
                if args.save:
                    test_set._save_predsegm(mask, save_root=SAVE_DIR, batch_size=batch_size, n_frame=n_frame, offset=i)

        # Evaluate
        print('Evaluation on %s-%s:'%(args.dataset, args.split))
        Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
        Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
        Confidence = np.concatenate(ap_eval_meter['Confidence'])
        N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
        AP = calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot='True')
        print('AveragePrecision@50:', AP)
        PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
        print('PanopticQuality@50:', PQ, 'F1-score@50:', F1, 'Prec@50:', Pre, 'Recall@50:', Rec)
        eval_avg = eval_meter.get_mean_loss_dict()
        print(eval_avg)