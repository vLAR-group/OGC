import torch
import torch.nn as nn

from pointnet2.pointnet2 import *


class ChamferLoss(nn.Module):
    """
    Chamfer distance.
    """
    def __init__(self, loss_norm=2):
        super().__init__()
        self.loss_norm = loss_norm

    def forward(self, pc1, pc2, flow):
        """
        :param pc1: (B, N, 3) torch.Tensor.
        :param pc2: (B, N, 3) torch.Tensor.
        :param flow: (B, N, 3) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        pc2 = pc2.contiguous()
        pc2_t = pc2.transpose(1, 2).contiguous()
        pc1 = (pc1 + flow).contiguous()
        pc1_t = pc1.transpose(1, 2).contiguous()

        _, idx = knn(1, pc1, pc2)
        nn1 = grouping_operation(pc2_t, idx.detach()).squeeze(-1)
        dist1 = (pc1_t - nn1).norm(p=self.loss_norm, dim=1)
        _, idx = knn(1, pc2, pc1)
        nn2 = grouping_operation(pc1_t, idx.detach()).squeeze(-1)
        dist2 = (pc2_t - nn2).norm(p=self.loss_norm, dim=1)
        loss = (dist1 + dist2).mean()
        return loss
    

class KnnLoss(nn.Module):
    """
    Part of the smooth loss by KNN.
    """
    def __init__(self, k, radius, loss_norm=1):
        super().__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc, flow):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param flow: (B, N, 3) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        pc = pc.contiguous()
        flow = flow.permute(0, 2, 1).contiguous()
        dist, idx = knn(self.k, pc, pc)
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]
        nn_flow = grouping_operation(flow, idx.detach())
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean()
        return loss


class BallQLoss(nn.Module):
    """
    Part of the smooth loss by ball query.
    """
    def __init__(self, k, radius, loss_norm=1):
        super().__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc, flow):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param flow: (B, N, 3) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        pc = pc.contiguous()
        flow = flow.permute(0, 2, 1).contiguous()
        idx = ball_query(self.radius, self.k, pc, pc)
        nn_flow = grouping_operation(flow, idx.detach())
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean()
        return loss


class SmoothLoss(nn.Module):
    """
    Enforce local smoothness of point-wise flow.
    """
    def __init__(self, w_knn, w_ball_q, knn_loss_params, ball_q_loss_params):
        super().__init__()
        self.knn_loss = KnnLoss(**knn_loss_params)
        self.ball_q_loss = BallQLoss(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc, flow):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param flow: (B, N, 3) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        loss = (self.w_knn * self.knn_loss(pc, flow)) + (self.w_ball_q * self.ball_q_loss(pc, flow))
        return loss


class UnsupervisedFlowStep3DLoss(nn.Module):
    def __init__(self, chamfer_loss, smooth_loss, weights=[0.75, 0.25], iters_w=[1.0]):
        super().__init__()
        self.chamfer_loss = chamfer_loss
        self.smooth_loss = smooth_loss
        self.w_chamfer, self.w_smooth = weights
        self.iters_w = iters_w

    def forward(self, pc1, pc2, flow_preds):
        """
        :param pc1 & pc2: (B, N, 3) torch.Tensor.
        :param flow_preds: [(B, N ,3), ...], list of torch.Tensor.
        """
        assert len(flow_preds) == len(self.iters_w)

        loss_dict = {}
        loss_arr = []
        for i in range(len(flow_preds)):
            flow_pred = flow_preds[i]
            chamfer_loss_i = self.chamfer_loss(pc1, pc2, flow_pred)
            loss_dict['chamfer_loss_#%d'%(i)] = chamfer_loss_i.item()
            smooth_loss_i = self.smooth_loss(pc1, flow_pred)
            loss_dict['smooth_loss_#%d'%(i)] = smooth_loss_i.item()
            loss_i = self.w_chamfer * chamfer_loss_i + self.w_smooth * smooth_loss_i
            loss_arr.append(self.iters_w[i] * loss_i)

        loss = sum(loss_arr)
        loss_dict['sum'] = loss.item()
        return loss, loss_dict


if __name__ == '__main__':
    chamfer_loss_params = {'loss_norm': 2}
    smooth_loss_params = {
        'w_knn': 3., 'w_ball_q': 1.,
        'knn_loss_params': {'k': 4,
                            'radius': 0.05,
                            'loss_norm': 1},
        'ball_q_loss_params': {'k': 8,
                               'radius': 0.1,
                               'loss_norm': 1}
    }
    weights = [0.75, 0.25]  # ['chamfer', 'smooth']
    chamfer_loss = ChamferLoss(**chamfer_loss_params).cuda()
    smooth_loss = SmoothLoss(**smooth_loss_params).cuda()
    criterion = UnsupervisedFlowStep3DLoss(chamfer_loss=chamfer_loss,
                                           smooth_loss=smooth_loss,
                                           weights=weights,
                                           iters_w=[1.0])

    pc1 = torch.randn(4, 512, 3).cuda()
    pc2 = torch.randn(4, 512, 3).cuda()
    flow_preds = [torch.randn(4, 512, 3).cuda()]
    loss, loss_dict = criterion(pc1, pc2, flow_preds)
    print (loss, loss_dict)