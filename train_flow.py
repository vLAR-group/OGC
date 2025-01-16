import os
import os.path as osp
import tqdm
import yaml
import argparse
import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from losses.flow_loss_unsup import ChamferLoss, SmoothLoss, UnsupervisedFlowStep3DLoss
from utils.pytorch_util import BNMomentumScheduler, save_checkpoint, checkpoint_state, AverageMeter, RunningAverageMeter


def epe_metric(gt_flow, flow_preds):
    """
    Monitor EPE3D of iterative predictions from FlowStep3D.
    :param gt_flow: (B, N, 3) torch.Tensor.
    :param flow_preds: [(B, N ,3), ...], list of torch.Tensor.
    """
    epe_dict = {}
    for i in range(len(flow_preds)):
        flow_pred = flow_preds[i].detach().cpu()
        epe_norm = torch.norm(flow_pred - gt_flow, dim=2)
        epe = epe_norm.mean()
        epe_dict['epe3d_#%d'%(i)] = epe.item()
    return epe_dict


class Trainer(object):
    def __init__(self,
                 flownet,
                 model_iters,
                 criterion,
                 optimizer,
                 exp_base,
                 lr_scheduler=None,
                 bnm_scheduler=None):
        self.flownet = flownet
        self.model_iters = model_iters
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.bnm_scheduler = bnm_scheduler

        self.exp_base = exp_base
        os.makedirs(exp_base, exist_ok=True)
        self.checkpoint_name, self.best_name = "current", "best"
        self.cur_epoch = 0
        self.training_best, self.eval_best = {}, {}
        log_dir = osp.join(exp_base, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.viz = SummaryWriter(log_dir)


    def _train_it(self, it, batch):
        self.flownet.train()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(it)
        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        # Forward
        with torch.set_grad_enabled(True):
            pcs, _, flows, _ = batch
            pcs = pcs.cuda()
            pc1, pc2 = pcs[:, 0].contiguous(), pcs[:, 1].contiguous()
            flow = flows[:, 0]

            flow_preds = self.flownet(pc1, pc2, pc1, pc2, iters=self.model_iters)
            loss, loss_dict = self.criterion(pc1, pc2, flow_preds)
            epe_dict = epe_metric(flow, flow_preds)
            loss_dict.update(epe_dict)

        # Backward
        try:
            loss.backward()
        except RuntimeError:
            return loss_dict

        for param in self.flownet.parameters():
            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                return loss_dict

        self.optimizer.step()
        return loss_dict


    def eval_epoch(self, val_loader):
        if self.flownet is not None:
            self.flownet.eval()

        eval_meter = AverageMeter()
        total_loss = 0.0
        count = 1.0

        with tqdm.tqdm(enumerate(val_loader, 0), total=len(val_loader), leave=False, desc='val') as tbar:
            for i, batch in tbar:
                with torch.set_grad_enabled(False):
                    pcs, _, flows, _ = batch
                    pcs = pcs.cuda()
                    pc1, pc2 = pcs[:, 0].contiguous(), pcs[:, 1].contiguous()
                    flow = flows[:, 0]

                    flow_preds = self.flownet(pc1, pc2, pc1, pc2, iters=self.model_iters)
                    loss, loss_dict = self.criterion(pc1, pc2, flow_preds)
                    epe_dict = epe_metric(flow, flow_preds)
                    loss_dict.update(epe_dict)

                total_loss += loss.item()
                count += 1
                eval_meter.append_loss(loss_dict)
                tbar.set_postfix(eval_meter.get_mean_loss_dict())

        return total_loss / count, eval_meter.get_mean_loss_dict()


    def train(self, n_epochs, train_loader, val_loader=None):
        it = 0
        best_loss = 1e10

        # Save initial model
        save_checkpoint(
            checkpoint_state(self.flownet), True,
            filename=osp.join(self.exp_base, self.checkpoint_name),
            bestname=osp.join(self.exp_base, self.best_name))

        with tqdm.trange(1, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            for epoch in tbar:
                train_meter = AverageMeter()
                train_running_meter = RunningAverageMeter(alpha=0.3)
                self.cur_epoch = epoch

                for batch in train_loader:
                    loss_dict = self._train_it(it, batch)
                    it += 1
                    pbar.update()
                    train_running_meter.append_loss(loss_dict)
                    pbar.set_postfix(train_running_meter.get_loss_dict())

                    # Monitor loss
                    tbar.refresh()
                    for loss_name, loss_val in loss_dict.items():
                        self.viz.add_scalar('train/'+loss_name, loss_val, global_step=it)
                    train_meter.append_loss(loss_dict)

                    if (it % len(train_loader)) == 0:
                        pbar.close()

                        # Accumulate train loss and metrics in the whole epoch
                        train_avg = train_meter.get_mean_loss_dict()
                        for meter_key, meter_val in train_avg.items():
                            self.viz.add_scalar('epoch_sum_train/'+meter_key, meter_val, global_step=epoch)

                        # Test on the validation set
                        if val_loader is not None:
                            val_loss, val_avg = self.eval_epoch(val_loader)
                            for meter_key, meter_val in val_avg.items():
                                self.viz.add_scalar('epoch_sum_val/'+meter_key, np.mean(meter_val), global_step=epoch)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(self.flownet),
                                is_best,
                                filename=osp.join(self.exp_base, self.checkpoint_name),
                                bestname=osp.join(self.exp_base, self.best_name))

                            # # Also save intermediate epochs
                            # save_checkpoint(
                            #     checkpoint_state(self.flownet),
                            #     is_best,
                            #     filename=osp.join(self.exp_base, 'epoch_%03d'%(self.cur_epoch)),
                            #     bestname=osp.join(self.exp_base, self.best_name))

                        pbar = tqdm.tqdm(
                            total=len(train_loader), leave=False, desc='train')
                        pbar.set_postfix(dict(total_it=it))

                    self.viz.flush()

        return best_loss


def lr_curve(it):
    return max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        args.lr_clip / args.lr,
    )


def bn_curve(it):
    if args.decay_step == -1:
        return args.bn_momentum
    else:
        return max(
            args.bn_momentum
            * args.bn_decay ** (int(it * args.batch_size / args.decay_step)),
            1e-2,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Fix the random seed
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Configuration for different dataset
    data_root = args.data['root']
    if args.dataset == 'sapien':
        from models.flownet_sapien import FlowStep3D
        from datasets.dataset_sapien import SapienDataset as TrainDataset
        data_root = osp.join(data_root, 'mbs-shapepart')
    elif args.dataset == 'ogcdr':
        from models.flownet_ogcdr import FlowStep3D
        from datasets.dataset_ogcdr import OGCDynamicRoomDataset as TrainDataset
    else:
        raise KeyError('Unrecognized dataset!')

    # Setup the network
    flownet = FlowStep3D(npoint=args.flownet['npoint'],
                         use_instance_norm=args.flownet['use_instance_norm'],
                         loc_flow_nn=args.flownet['loc_flow_nn'],
                         loc_flow_rad=args.flownet['loc_flow_rad'],
                         k_decay_fact=args.flownet['k_decay_fact']).cuda()

    # Only use adjacent frame pairs (Self-supervised training cannot handle large motions)
    view_sels = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]

    # Setup the dataset
    train_set = TrainDataset(data_root=data_root,
                             split='train',
                             view_sels=view_sels,
                             aug_transform=args.data['aug_transform'],
                             aug_transform_args=args.data['aug_transform_args'])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_set = TrainDataset(data_root=data_root,
                           split='val',
                           view_sels=view_sels,
                           aug_transform=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Setup the optimizer
    optimizer = optim.Adam(flownet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_curve)
    bnm_scheduler = BNMomentumScheduler(flownet, bn_lambda=bn_curve)

    # Setup the loss
    chamfer_loss = ChamferLoss(**args.loss['chamfer_loss_params'])
    smooth_loss = SmoothLoss(**args.loss['smooth_loss_params'])
    criterion = UnsupervisedFlowStep3DLoss(chamfer_loss=chamfer_loss,
                                           smooth_loss=smooth_loss,
                                           iters_w=args.loss['iters_w'],
                                           weights=args.loss['weights'])

    # Setup the trainer
    trainer = Trainer(flownet=flownet,
                      model_iters=args.model_iters,
                      criterion=criterion,
                      optimizer=optimizer,
                      exp_base=args.save_path,
                      lr_scheduler=lr_scheduler,
                      bnm_scheduler=bnm_scheduler)

    # Train
    trainer.train(args.epochs, train_loader, val_loader)