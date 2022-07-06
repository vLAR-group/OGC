import torch
import torch.nn as nn
import numpy as np
import shutil
from tensorboardX import SummaryWriter
from collections import OrderedDict


class RunningAverageMeter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: loss_val})
            else:
                old_mean = self.loss_dict[loss_name]
                self.loss_dict[loss_name] = self.alpha * old_mean + (1 - self.alpha) * loss_val

    def get_loss_dict(self):
        return {k: v for k, v in self.loss_dict.items()}


class AverageMeter:
    def __init__(self):
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val, 1]})
            else:
                self.loss_dict[loss_name][0] += loss_val
                self.loss_dict[loss_name][1] += 1

    def get_mean_loss(self):
        all_loss_val = 0.0
        all_loss_count = 0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_val += loss_val
            all_loss_count += loss_count
        return all_loss_val / (all_loss_count / len(self.loss_dict))

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            loss_dict[loss_name] = loss_val / loss_count
        return loss_dict

    def get_printable(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_sum += loss_val / loss_count
            text += "(%s:%.4f) " % (loss_name, loss_val / loss_count)
        text += " sum = %.4f" % all_loss_sum
        return text


class TensorboardViz:
    def __init__(self, logdir):
        self.logdir = logdir
        self.writter = SummaryWriter(self.logdir)

    def update(self, mode, it, eval_dict):
        self.writter.add_scalars(mode, eval_dict, global_step=it)

    def flush(self):
        self.writter.flush()


def checkpoint_state(model):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    return {'model_state': model_state}


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                          nn.GroupNorm)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self,
                 model,
                 bn_lambda,
                 last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(
                type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_momentum(self):
        return self.lmbd(self.last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))