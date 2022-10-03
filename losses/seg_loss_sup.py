import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid=None):
        """
        :param pred: (B, N, K) torch.Tensor.
        :param target: (B, N, K) torch.Tensor, the same as pred.
        :param valid: (B, N) torch.Tensor, if available.
        :return:
            loss: () torch.Tensor.
        """
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        if valid is not None:
            valid = valid.unsqueeze(2)
            loss = loss * valid
        return loss.mean()

    def match_cost(self, pred, target, valid=None):
        """
        :param pred: (B, N, K, K) torch.Tensor.
        :param target: (B, N, K, K) torch.Tensor.
        :param valid: (B, N) torch.Tensor, if available.
        :return:
            loss: (B, K, K) torch.Tensor.
        """
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        if valid is not None:
            valid = valid.unsqueeze(2).unsqueeze(3)
            loss = loss * valid
        return loss.mean(1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target, valid=None):
        """
        :param pred: (B, N, K) torch.Tensor.
        :param target: (B, N, K) torch.Tensor, the same as pred.
        :param valid: (B, N) torch.Tensor, if available.
        :return:
            loss: () torch.Tensor.
        """
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = target * pred + (1 - target) * (1 - pred)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if valid is not None:
            valid = valid.unsqueeze(2)
            loss = loss * valid
        return loss.mean()

    def match_cost(self, pred, target, valid=None):
        """
        :param pred: (B, N, K, K) torch.Tensor.
        :param target: (B, N, K, K) torch.Tensor.
        :param valid: (B, N) torch.Tensor, if available.
        :return:
            loss: (B, K, K) torch.Tensor.
        """
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = target * pred + (1 - target) * (1 - pred)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if valid is not None:
            valid = valid.unsqueeze(2).unsqueeze(3)
            loss = loss * valid
        return loss.mean(1)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid=None):
        """
        :param pred: (B, N, K) torch.Tensor.
        :param target: (B, N, K) torch.Tensor, the same as pred.
        :param valid: (B, N) torch.Tensor, if available.
        :return:
            loss: () torch.Tensor.
        """
        if valid is not None:
            valid = valid.unsqueeze(2)
            numerator = (2 * (pred * target) * valid).sum(1)
            denominator = (pred * valid).sum(1) + (target * valid).sum(1)
        else:
            numerator = 2 * (pred * target).sum(1)
            denominator = pred.sum(1) + target.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    def match_cost(self, pred, target, valid=None):
        """
        :param pred: (B, N, K, K) torch.Tensor.
        :param target: (B, N, K, K) torch.Tensor.
        :param valid: (B, N) torch.Tensor, if available.
        :return:
            loss: (B, K, K) torch.Tensor.
        """
        if valid is not None:
            valid = valid.unsqueeze(2).unsqueeze(3)
            numerator = (2 * (pred * target) * valid).sum(1)
            denominator = (pred * valid).sum(1) + (target * valid).sum(1)
        else:
            numerator = 2 * (pred * target).sum(1)
            denominator = pred.sum(1) + target.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()


def match_mask_by_cost(cost):
    """
    :param cost: (B, K, K) torch.Tensor.
    :return:
        perm: (B, K, K) torch.Tensor, permutation for minimizing cost.
    """
    n_batch, _, n_object = cost.size()
    perm = []
    for b in range(n_batch):
        cost_b = cost[b].cpu().numpy()
        _, col_ind = linear_sum_assignment(cost_b, maximize=False)
        perm.append(col_ind)
    perm = torch.from_numpy(np.stack(perm, 0))
    perm = torch.eye(n_object, dtype=torch.float32, device=cost.device)[perm]
    return perm


class SupervisedMaskLoss(nn.Module):
    def __init__(self, ce_loss, dice_loss, weights=[2.0, 0.1]):
        super().__init__()
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.w_ce, self.w_dice = weights

    def forward(self, mask, gt_mask, valid=None):
        """
        :param mask: (B, N, K) torch.Tensor.
        :param gt_mask: (B, N, K) torch.Tensor.
        :param valid: (B, N) torch.Tensor, if available.
        """
        loss_dict = {}
        n_object = mask.shape[-1]

        mask_rep = mask.unsqueeze(3).repeat(1, 1, 1, n_object).detach()
        gt_mask_rep = gt_mask.unsqueeze(2).repeat(1, 1, n_object, 1)

        # Match objects in predictions and GT with Hungarian
        match_cost_ce = self.ce_loss.match_cost(mask_rep, gt_mask_rep, valid)
        match_cost_dice = self.dice_loss.match_cost(mask_rep, gt_mask_rep, valid)
        match_cost = self.w_ce * match_cost_ce + self.w_dice * match_cost_dice
        perm = match_mask_by_cost(match_cost)
        # Reorder objects in GT mask
        gt_mask = torch.einsum('bij,bnj->bni', perm, gt_mask).detach()

        l_ce = self.ce_loss(mask, gt_mask, valid)
        loss_dict['cross_entropy'] = l_ce.item()
        l_dice = self.dice_loss(mask, gt_mask, valid)
        loss_dict['dice'] = l_dice.item()

        loss = self.w_ce * l_ce + self.w_dice * l_dice
        loss_dict['sum'] = loss.item()
        return loss, loss_dict