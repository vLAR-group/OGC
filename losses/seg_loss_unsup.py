import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from pointnet2.pointnet2 import *


def fit_motion_svd_batch(pc1, pc2, mask=None):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask: (B, N) torch.Tensor.
    :return:
        R_base: (B, 3, 3) torch.Tensor.
        t_base: (B, 3) torch.Tensor.
    """
    n_batch, n_point, _ = pc1.size()

    if mask is None:
        pc1_mean = torch.mean(pc1, dim=1, keepdim=True)   # (B, 1, 3)
        pc2_mean = torch.mean(pc2, dim=1, keepdim=True)   # (B, 1, 3)
    else:
        pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)   # (B, 3)
        pc1_mean.unsqueeze_(1)
        pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
        pc2_mean.unsqueeze_(1)

    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    if mask is None:
        S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
    else:
        S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask).bmm(pc2_centered))

    # If mask is not well-defined, S will be ill-posed.
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.zeros((n_batch, 3), device=pc1.device)

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        # Correct reflection matrix to rotation matrix
        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
        t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)

        R_base[valid_batches] = R
        t_base[valid_batches] = t

    return R_base, t_base


class DynamicLoss(nn.Module):
    """
    Enforce the rigid transformation estimated from object masks to explain the per-point flow.
    """
    def __init__(self, loss_norm=2):
        super().__init__()
        self.loss_norm = loss_norm

    def forward(self, pc, mask, flow):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :param flow: (B, N, 3) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        n_batch, n_point, n_object = mask.size()
        pc2 = pc + flow
        mask = mask.transpose(1, 2).reshape(n_batch * n_object, n_point)
        pc_rep = pc.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
        pc2_rep = pc2.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

        # Estimate the rigid transformation
        object_R, object_t = fit_motion_svd_batch(pc_rep, pc2_rep, mask)

        # Apply the estimated rigid transformation onto point cloud
        pc_transformed = torch.einsum('bij,bnj->bni', object_R, pc_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc_transformed = pc_transformed.reshape(n_batch, n_object, n_point, 3).detach()
        mask = mask.reshape(n_batch, n_object, n_point)

        # Measure the discrepancy of per-point flow
        mask = mask.unsqueeze(-1)
        pc_transformed = (mask * pc_transformed).sum(1)
        loss = (pc_transformed - pc2).norm(p=self.loss_norm, dim=-1)
        return loss.mean()


class KnnLoss(nn.Module):
    """
    Part of the smooth loss by KNN.
    """
    def __init__(self, k, radius, cross_entropy=False, loss_norm=1, **kwargs):
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        mask = mask.permute(0, 2, 1).contiguous()
        dist, idx = knn(self.k, pc, pc)
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]
        nn_mask = grouping_operation(mask, idx.detach())
        if self.cross_entropy:
            mask = mask.unsqueeze(3).repeat(1, 1, 1, self.k).detach()
            loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
        else:
            loss = (mask.unsqueeze(3) - nn_mask).norm(p=self.loss_norm, dim=1).mean(dim=-1)
        return loss.mean()


class BallQLoss(nn.Module):
    """
    Part of the smooth loss by ball query.
    """
    def __init__(self, k, radius, cross_entropy=False, loss_norm=1, **kwargs):
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        mask = mask.permute(0, 2, 1).contiguous()
        idx = ball_query(self.radius, self.k, pc, pc)
        nn_mask = grouping_operation(mask, idx.detach())
        if self.cross_entropy:
            mask = mask.unsqueeze(3).repeat(1, 1, 1, self.k).detach()
            loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
        else:
            loss = (mask.unsqueeze(3) - nn_mask).norm(p=self.loss_norm, dim=1).mean(dim=-1)
        return loss.mean()


class SmoothLoss(nn.Module):
    """
    Enforce local smoothness of object mask.
    """
    def __init__(self, w_knn, w_ball_q, knn_loss_params, ball_q_loss_params):
        super().__init__()
        self.knn_loss = KnnLoss(**knn_loss_params)
        self.ball_q_loss = BallQLoss(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc, mask):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        loss = (self.w_knn * self.knn_loss(pc, mask)) + (self.w_ball_q * self.ball_q_loss(pc, mask))
        return loss


def interpolate_mask_by_flow(pc1, pc2, mask1, flow1, k=1):
    """
    Interpolate the object mask of pc2 from its K-nearest neighbors' object mask in warped_pc1.
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask1: (B, N, K) torch.Tensor.
    :param flow1: (B, N, 3) torch.Tensor.
    :return:
        mask2_interpolated: (B, N, K) torch.Tensor.
    """
    warped_pc1 = pc1 + flow1
    dist, idx = knn(k, pc2, warped_pc1)

    # Find K-nearest neighbors' object mask
    mask1 = mask1.transpose(1, 2).contiguous()
    mask2_interpolated = grouping_operation(mask1, idx.detach())

    # Interpolate from K-nearest neighbors' object mask
    if k == 1:
        mask2_interpolated = mask2_interpolated.squeeze(-1)
    else:
        dist = dist.clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm
        mask2_interpolated = torch.sum(weight.unsqueeze(1) * mask2_interpolated, dim=-1)
    mask2_interpolated = mask2_interpolated.transpose(1, 2)
    return mask2_interpolated


def match_mask_by_iou(mask1, mask2):
    """
    Match individual objects in two object masks by Hungarian algorithm.
    :param mask1: (B, N, K) torch.Tensor.
    :param mask2: (B, N, K) torch.Tensor.
    :return:
        perm: (B, K, K) torch.Tensor, permutation for alignment.
    """
    # Transform soft mask to hard segmentation (one-hot)
    n_batch, _, n_object = mask1.size()
    segm_pred1 = mask1.argmax(-1).detach()
    segm_pred2 = mask2.argmax(-1).detach()
    segm_pred1 = torch.eye(n_object, dtype=torch.float32,
                           device=segm_pred1.device)[segm_pred1]
    segm_pred2 = torch.eye(n_object, dtype=torch.float32,
                           device=segm_pred2.device)[segm_pred2]

    # Match according to IoU
    intersection = torch.einsum('bng,bnp->bgp', segm_pred1, segm_pred2)     # (B, K, K)
    union = torch.sum(segm_pred1, dim=1).unsqueeze(-1) + torch.sum(segm_pred2, dim=1, keepdim=True) - intersection  # (B, K, K)
    iou = intersection / union.clamp(1e-10)
    perm = []
    for b in range(n_batch):
        iou_score = iou[b].cpu().numpy()
        _, col_ind = linear_sum_assignment(iou_score, maximize=True)
        perm.append(col_ind)
    perm = torch.from_numpy(np.stack(perm, 0))
    perm = torch.eye(n_object, dtype=torch.float32, device=segm_pred1.device)[perm]
    return perm


class InvarianceLoss(nn.Module):
    """
    Minimize the difference between matched per-point segmentation.
    """
    def __init__(self, cross_entropy=False, loss_norm=2):
        super().__init__()
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def distance(self, mask1, mask2):
        """
        :param mask1: (B, N, K) torch.Tensor, prediction.
        :param mask2: (B, N, K) torch.Tensor, target.
        :return:
            loss: () torch.Tensor.
        """
        if self.cross_entropy:
            loss = F.binary_cross_entropy(mask1, mask2, reduction='none').sum(dim=1)
        else:
            loss = (mask1 - mask2).norm(p=self.loss_norm, dim=-1)
        return loss.mean()

    def forward(self, mask1, mask2):
        """
        :param mask1: (B, N, K) torch.Tensor.
        :param mask2: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        # Aligh the object ordering in two views
        perm2 = match_mask_by_iou(mask1, mask2)
        target_mask1 = torch.einsum('bij,bnj->bni', perm2, mask2).detach()
        perm1 = match_mask_by_iou(mask2, mask1)
        target_mask2 = torch.einsum('bij,bnj->bni', perm1, mask1).detach()

        # Enforce the invariance
        loss = self.distance(mask1, target_mask1) + self.distance(mask2, target_mask2)
        return loss


class EntropyLoss(nn.Module):
    """
    Push object mask towards one-hot.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mask, epsilon=1e-5):
        """
        :param mask: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        loss = - (mask * torch.log(mask.clamp(epsilon))).sum(dim=-1)
        return loss.mean()


class RankLoss(nn.Module):
    """
    Push object mask towards low-rank (avoid over-segmentation).
    """
    def __init__(self):
        super().__init__()

    def forward(self, mask):
        """
        :param mask: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        loss = mask.norm(p='nuc', dim=(1, 2))
        return loss.mean()


class UnsupervisedOGCLoss(nn.Module):
    def __init__(self,
                 dynamic_loss, smooth_loss, invariance_loss, entropy_loss, rank_loss,
                 weights=[10.0, 0.1, 0.1], start_steps=[0, 0, 0]):
        super().__init__()
        self.dynamic_loss = dynamic_loss
        self.smooth_loss = smooth_loss
        self.invariance_loss = invariance_loss
        self.w_dynamic, self.w_smooth, self.w_invariance = weights
        self.start_step_dynamic, self.start_step_smooth, self.start_step_invariance = start_steps

        # Entropy & Rank not participate in BP, just for monitoring
        self.entropy_loss = entropy_loss
        self.rank_loss = rank_loss

    def step_lossw(self, it, weight, start_step=0):
        if it < start_step:
            return 0
        else:
            return weight

    def forward(self, pcs, masks, flows, step_w=False, it=0, aug_transform=False):
        """
        :param pcs: list of torch.Tensor, [2 * (B, N, 3)] or [4 * (B, N, 3)]
        :param masks: list of torch.Tensor, [2 * (B, N, K)] or [4 * (B, N, K)]
        :param flows: list of torch.Tensor, [2 * (B, N, 3)] or [4 * (B, N, 3)]
        """
        assert len(pcs) == len(masks) == len(flows), "Inconsistent number of frames!"

        loss_arr = []
        loss_dict = {}
        if aug_transform:
            pc1, pc2, pc3, pc4 = pcs
            mask1, mask2, mask3, mask4 = masks
            flow1, flow2, flow3, flow4 = flows
        else:
            pc1, pc2 = pcs
            mask1, mask2 = masks
            flow1, flow2 = flows

        # 1. Rigid loss
        l_dynamic = self.dynamic_loss(pc1, mask1, flow1) + self.dynamic_loss(pc2, mask2, flow2)
        if aug_transform:
            l_dynamic += self.dynamic_loss(pc3, mask3, flow3) + self.dynamic_loss(pc4, mask4, flow4)
            l_dynamic = 0.5 * l_dynamic
        loss_dict['dynamic'] = l_dynamic.item()
        if step_w:
            w = self.step_lossw(it, weight=self.w_dynamic, start_step=self.start_step_dynamic)
        else:
            w = self.w_dynamic
        loss_arr.append(w * l_dynamic)

        # 2. Smooth loss
        l_smooth = self.smooth_loss(pc1, mask1) + self.smooth_loss(pc2, mask2)
        if aug_transform:
            l_smooth += self.smooth_loss(pc3, mask3) + self.smooth_loss(pc4, mask4)
            l_smooth = 0.5 * l_smooth
        loss_dict['smooth'] = l_smooth.item()
        if step_w:
            w = self.step_lossw(it, weight=self.w_smooth, start_step=self.start_step_smooth)
        else:
            w = self.w_smooth
        loss_arr.append(w * l_smooth)

        # 3. Invariance loss
        if aug_transform:
            l_invariance = self.invariance_loss(mask1, mask3) + self.invariance_loss(mask2, mask4)
            loss_dict['invariance'] = l_invariance.item()
            if step_w:
                w = self.step_lossw(it, weight=self.w_invariance, start_step=self.start_step_invariance)
            else:
                w = self.w_invariance
            loss_arr.append(w * l_invariance)
        else:
            loss_dict['invariance'] = 0

        # 4. Entropy (for monitoring only)
        l_entropy = self.entropy_loss(mask1) + self.entropy_loss(mask2)
        if aug_transform:
            l_entropy += self.entropy_loss(mask3) + self.entropy_loss(mask4)
            l_entropy = 0.5 * l_entropy
        loss_dict['entropy'] = l_entropy.item()

        # 5. Rank (for monitoring only)
        l_rank = self.rank_loss(mask1) + self.rank_loss(mask2)
        if aug_transform:
            l_rank += self.rank_loss(mask3) + self.rank_loss(mask4)
            l_rank = 0.5 * l_rank
        loss_dict['rank'] = l_rank.item()

        loss = sum(loss_arr)
        loss_dict['sum'] = loss.item()
        return loss, loss_dict