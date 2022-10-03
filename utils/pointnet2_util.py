import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.pointnet2 import *
from utils.nn_util import SharedMLP


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None, return_inds=False):
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.npoint is not None:
            new_inds = furthest_point_sample(xyz, self.npoint).long()
        new_xyz = (
            gather_nd(
                xyz_flipped, new_inds, t=True
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            # else xyz_flipped.new_zeros((xyz_flipped.size(0), 1, 3))     # This matches original implementation.
            else None     # This matches original implementation.
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )[0]  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        if return_inds:
            return new_xyz, torch.cat(new_features_list, dim=1), new_inds
        else:
            return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    def __init__(self, npoint, radii, nsamples, mlps, bn, use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    def __init__(
        self, mlp, npoint, radius, nsample, bn, use_xyz=True
            # Note: if npoint=radius=nsample=None then will be gather all operation.
    ):
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    def __init__(self, mlp, bn):
        super(PointnetFPModule, self).__init__()
        self.mlp = SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        if known is not None:
            dist, idx = three_nn(unknown.contiguous(), known.contiguous())
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(
                known_feats.contiguous(), idx.contiguous(), weight.contiguous()
            )
        else:
            interpolated_feats = known_feats.expand(
                *(list(known_feats.size()[0:2]) + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)