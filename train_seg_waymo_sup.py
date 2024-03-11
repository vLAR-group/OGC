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

from losses.seg_loss_sup import CrossEntropyLoss, FocalLoss, DiceLoss, SupervisedMaskLoss
from metrics.seg_metric import accumulate_eval_results, calculate_PQ_F1
from utils.pytorch_util import BNMomentumScheduler, save_checkpoint, checkpoint_state, AverageMeter, RunningAverageMeter


class Trainer(object):
    def __init__(self,
                 segnet,
                 criterion,
                 optimizer,
                 exp_base,
                 lr_scheduler=None,
                 bnm_scheduler=None):
        self.segnet = segnet
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.bnm_scheduler = bnm_scheduler

        self.exp_base = exp_base
        os.makedirs(exp_base, exist_ok=True)
        self.checkpoint_name, self.best_name = "current", "best"
        self.cur_epoch = 0
        self.training_best, self.eval_best = {}, {}
        self.viz = SummaryWriter(osp.join(exp_base, 'log'))


    def _train_it(self, it, batch, aug_transform=False):
        self.segnet.train()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(it)
        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        # Forward
        with torch.set_grad_enabled(True):
            pcs, segms, valids = batch

            pc = pcs[:, 0].contiguous().cuda()
            gt_mask = segms[:, 0].contiguous().cuda()
            valid = valids[:, 0].contiguous().cuda()
            mask = self.segnet(pc, pc)

            loss, loss_dict = self.criterion(mask, gt_mask, valid)

        segm = gt_mask.argmax(2).cpu()
        mask = mask.detach().cpu()

        # Backward
        try:
            loss.backward()
        except RuntimeError:
            return loss_dict, segm, mask

        for param in self.segnet.parameters():
            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                return loss_dict, segm, mask

        self.optimizer.step()
        return loss_dict, segm, mask


    def eval_epoch(self, d_loader):
        if self.segnet is not None:
            self.segnet.eval()

        eval_meter = AverageMeter()
        total_loss = 0.0
        count = 1.0

        ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
        with tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val') as tbar:
            for i, batch in tbar:
                with torch.set_grad_enabled(False):
                    pcs, segms, valids = batch

                    pc = pcs[:, 0].contiguous().cuda()
                    gt_mask = segms[:, 0].contiguous().cuda()
                    valid = valids[:, 0].contiguous().cuda()
                    mask = self.segnet(pc, pc)

                    loss, loss_dict = self.criterion(mask, gt_mask, valid)

                total_loss += loss.item()
                count += 1
                eval_meter.append_loss(loss_dict)
                tbar.set_postfix(eval_meter.get_mean_loss_dict())

                segm = gt_mask.argmax(2).cpu()
                mask = mask.detach().cpu()

                Pred_IoU, Pred_Matched, _, N_GT_Inst = accumulate_eval_results(segm, mask)
                ap_eval_meter['Pred_IoU'].append(Pred_IoU)
                ap_eval_meter['Pred_Matched'].append(Pred_Matched)
                ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

        return total_loss / count, eval_meter.get_mean_loss_dict(), ap_eval_meter


    def train(self, n_epochs, train_loader, test_loader=None):
        it = 0
        best_loss = 1e10
        aug_transform = False

        # Save init model.
        save_checkpoint(
            checkpoint_state(self.segnet), True,
            filename=osp.join(self.exp_base, self.checkpoint_name),
            bestname=osp.join(self.exp_base, self.best_name))

        with tqdm.trange(1, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            for epoch in tbar:
                train_meter = AverageMeter()
                train_running_meter = RunningAverageMeter(alpha=0.3)
                self.cur_epoch = epoch

                ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
                for batch in train_loader:
                    loss_dict, segm, mask = self._train_it(it, batch, aug_transform=aug_transform)
                    it += 1
                    pbar.update()
                    train_running_meter.append_loss(loss_dict)
                    pbar.set_postfix(train_running_meter.get_loss_dict())

                    # Monitor loss
                    tbar.refresh()
                    for loss_name, loss_val in loss_dict.items():
                        self.viz.add_scalar('train/'+loss_name, loss_val, global_step=it)
                    train_meter.append_loss(loss_dict)

                    # Monitor by quantitative evaluation metrics
                    Pred_IoU, Pred_Matched, _, N_GT_Inst = accumulate_eval_results(segm, mask)
                    ap_eval_meter['Pred_IoU'].append(Pred_IoU)
                    ap_eval_meter['Pred_Matched'].append(Pred_Matched)
                    ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

                    if (it % len(train_loader)) == 0:
                        pbar.close()

                        # Accumulate train loss and metrics in the epoch
                        train_avg = train_meter.get_mean_loss_dict()
                        for meter_key, meter_val in train_avg.items():
                            self.viz.add_scalar('epoch_sum_train/' + meter_key, meter_val, global_step=epoch)
                        Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
                        Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
                        N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
                        PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
                        self.viz.add_scalar('epoch_sum_train/PQ@50:', PQ, global_step=epoch)
                        self.viz.add_scalar('epoch_sum_train/F1@50:', F1, global_step=epoch)
                        self.viz.add_scalar('epoch_sum_train/Pre@50', Pre, global_step=epoch)
                        self.viz.add_scalar('epoch_sum_train/Rec@50', Rec, global_step=epoch)

                        # Test on the validation set
                        if test_loader is not None:
                            val_loss, val_avg, ap_eval_meter = self.eval_epoch(test_loader)
                            for meter_key, meter_val in val_avg.items():
                                self.viz.add_scalar('epoch_sum_val/'+meter_key, np.mean(val_avg[meter_key]), global_step=epoch)
                            Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
                            Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
                            N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
                            PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
                            self.viz.add_scalar('epoch_sum_val/PQ@50:', PQ, global_step=epoch)
                            self.viz.add_scalar('epoch_sum_val/F1@50:', F1, global_step=epoch)
                            self.viz.add_scalar('epoch_sum_val/Pre@50', Pre, global_step=epoch)
                            self.viz.add_scalar('epoch_sum_val/Rec@50', Rec, global_step=epoch)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(self.segnet),
                                is_best,
                                filename=osp.join(self.exp_base, self.checkpoint_name),
                                bestname=osp.join(self.exp_base, self.best_name))

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
    parser.add_argument('--round', type=int, default=0, help='Which round of iterative optimization')

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
    from models.segnet_kitti import MaskFormer3D
    from datasets.dataset_waymo_singleframe import WaymoOpenDataset as TrainDataset

    # Setup the network
    segnet = MaskFormer3D(n_slot=args.segnet['n_slot'],
                          n_point=args.segnet['n_point'],
                          use_xyz=args.segnet['use_xyz'],
                          n_transformer_layer=args.segnet['n_transformer_layer'],
                          transformer_embed_dim=args.segnet['transformer_embed_dim'],
                          transformer_input_pos_enc=args.segnet['transformer_input_pos_enc']).cuda()

    # Setup the dataset
    ignore_class_ids = [2, 3]
    train_set = TrainDataset(data_root=data_root,
                             mapping_path=args.data['train_mapping'],
                             downsampled=True,
                             select_frame=args.data['train_select_frame'],
                             aug_transform=args.data['aug_transform'],
                             aug_transform_args=args.data['aug_transform_args'],
                             decentralize=args.data['decentralize'],
                             onehot_label=True,
                             max_n_object=args.segnet['n_slot'],
                             ignore_class_ids=ignore_class_ids,
                             ignore_npoint_thresh=args.ignore_npoint_thresh)
    val_set = TrainDataset(data_root=data_root,
                           mapping_path=args.data['val_mapping'],
                           downsampled=True,
                           select_frame=args.data['val_select_frame'],
                           decentralize=args.data['decentralize'],
                           onehot_label=True,
                           max_n_object=args.segnet['n_slot'],
                           ignore_class_ids=ignore_class_ids,
                           ignore_npoint_thresh=args.ignore_npoint_thresh)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Setup the optimizer
    optimizer = optim.Adam(segnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_curve)
    bnm_scheduler = BNMomentumScheduler(segnet, bn_lambda=bn_curve)

    # Setup the loss
    if args.loss['use_focal']:
        ce_loss = FocalLoss(**args.loss['focal_loss_params'])
    else:
        ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss()
    criterion = SupervisedMaskLoss(ce_loss, dice_loss,
                                   weights=args.loss['weights'])

    # Setup the trainer
    trainer = Trainer(segnet=segnet,
                      criterion=criterion,
                      optimizer=optimizer,
                      exp_base=args.save_path,
                      lr_scheduler=lr_scheduler,
                      bnm_scheduler=bnm_scheduler)

    # Train
    trainer.train(args.epochs, train_loader, val_loader)