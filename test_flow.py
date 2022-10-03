import os
import os.path as osp
import tqdm
import yaml
import argparse
import json

import torch
from torch.utils.data import DataLoader

from metrics.flow_metric import eval_flow
from utils.pytorch_util import AverageMeter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--test_batch_size', type=int, default=48, help='Batch size in testing')
    parser.add_argument('--test_model_iters', type=int, default=4, help='Number of FlowStep3D model unrolling iterations in testing')
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
        from models.flownet_sapien import FlowStep3D
        from datasets.dataset_sapien import SapienDataset as TestDataset
        if args.split == 'test':
            data_root = osp.join(data_root, 'mbs-sapien')
        else:
            data_root = osp.join(data_root, 'mbs-shapepart')
        epe_norm_thresh = 0.01
    elif args.dataset == 'ogcdr':
        from models.flownet_ogcdr import FlowStep3D
        from datasets.dataset_ogcdr import OGCDynamicRoomDataset as TestDataset
        epe_norm_thresh = 0.01
    else:
        raise KeyError('Unrecognized dataset!')

    # Setup the network
    flownet = FlowStep3D(npoint=args.flownet['npoint'],
                         use_instance_norm=args.flownet['use_instance_norm'],
                         loc_flow_nn=args.flownet['loc_flow_nn'],
                         loc_flow_rad=args.flownet['loc_flow_rad'],
                         k_decay_fact=0.5).cuda()

    # Load the trained model weights
    # weight_path = osp.join(args.save_path, 'epoch_010.pth.tar')
    weight_path = osp.join(args.save_path, 'best.pth.tar')
    flownet.load_state_dict(torch.load(weight_path)['model_state'])
    flownet.cuda().eval()
    print('Loaded weights from', weight_path)

    # Setup the dataset
    view_sels = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
    test_set = TestDataset(data_root=data_root,
                           split=args.split,
                           view_sels=view_sels)
    batch_size = args.test_batch_size
    n_frame = len(view_sels)

    # Save flow predictions
    if args.save:
        assert batch_size % n_frame == 0, \
            'Frame pairs of one scene should be in the same batch, otherwise very inconvenient for saving!'
        # Path to save flow predictions
        # SAVE_DIR = osp.join(data_root, 'flow_preds/flowstep3d_epoch010')
        SAVE_DIR = osp.join(data_root, 'flow_preds/flowstep3d')
        os.makedirs(SAVE_DIR, exist_ok=True)
        # Write information about "view_sel" into a meta file
        SAVE_META = SAVE_DIR + '.json'
        with open(SAVE_META, 'w') as f:
            json.dump({'view_sel': view_sels}, f)


    # Iterate over the dataset
    eval_meter = AverageMeter()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
        for i, batch in tbar:
            pcs, _, flows, _ = batch
            pcs = pcs.cuda()
            pc1, pc2 = pcs[:, 0].contiguous(), pcs[:, 1].contiguous()
            flow = flows[:, 0]      # Groundtruth flow

            # Forward inference
            flow_preds = flownet(pc1, pc2, pc1, pc2, iters=args.test_model_iters)
            flow_pred = flow_preds[-1].detach().cpu()

            # Evaluate
            epe, acc_strict, acc_relax, outlier = eval_flow(flow, flow_pred, epe_norm_thresh=epe_norm_thresh)
            eval_meter.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})

            # Save
            if args.save:
                test_set._save_predflow(flow_pred, save_root=SAVE_DIR, batch_size=batch_size, n_frame=n_frame, offset=i)

    # Accumulate evaluation results
    eval_avg = eval_meter.get_mean_loss_dict()
    print('Evaluation on %s-%s:'%(args.dataset, args.split), eval_avg)