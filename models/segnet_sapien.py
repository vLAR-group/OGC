import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn_util import Seq
from utils.pointnet2_util import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
from utils.transformer_util import MaskFormerHead

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}


class MaskFormer3D(nn.Module):
    """
    A 3D object segmentation network, combing PointNet++ and MaskFormer.
    """
    def __init__(self,
                 n_slot,
                 n_point=512,
                 use_xyz=True,
                 bn=BN_CONFIG,
                 n_transformer_layer=2,
                 transformer_embed_dim=256,
                 transformer_input_pos_enc=False):
        super().__init__()

        # PointNet++ encoder & decoder to extract point embeddings
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=int(n_point / 2), radii=[0.1, 0.2],
                nsamples=[64, 64], mlps=[[3, 64, 64, 64], [3, 64, 64, 128]], use_xyz=use_xyz,
                bn=bn
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=int(n_point / 4), radius=0.4,
                nsample=64, mlp=[64 + 128, 128, 128, 256], use_xyz=use_xyz,
                bn=bn
            )
        )
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 64], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64 + 128, 256, 128], bn=bn))

        # MaskFormer head
        self.MF_head = MaskFormerHead(
            n_slot=n_slot, input_dim=256, n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim, transformer_n_head=8,
            transformer_hidden_dim=transformer_embed_dim, input_pos_enc=transformer_input_pos_enc
        )
        self.object_mlp = Seq(transformer_embed_dim).conv1d(transformer_embed_dim, bn=bn).conv1d(64, activation=None)


    def forward(self, pc, point_feats):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param point_feats: (B, N, 3) torch.Tensor.
        :return:
            mask: (B, N, K) torch.Tensor.
        """
        # Extract point embeddings with PointNet++ encoder & decoder
        l_pc, l_feats = [pc], [point_feats.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_pc, li_feats = self.SA_modules[i](l_pc[i], l_feats[i])
            l_pc.append(li_pc)
            l_feats.append(li_feats)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_feats[i - 1] = self.FP_modules[i](
                l_pc[i - 1], l_pc[i], l_feats[i - 1], l_feats[i]
            )

        # Extract object embeddings with MaskFormer head
        slot = self.MF_head(l_feats[-1].transpose(1, 2), l_pc[-1])     # (B, K, D)
        slot = self.object_mlp(slot.transpose(1, 2))      # (B, D, K)

        # Obtain mask by dot-product
        mask = torch.einsum('bdn,bdk->bnk',
                            F.normalize(l_feats[0], dim=1),
                            F.normalize(slot, dim=1)) / 0.05
        mask = mask.softmax(dim=-1)
        return mask


# Test the network implementation
if __name__ == '__main__':
    segnet = MaskFormer3D(n_slot=8,
                          use_xyz=True,
                          n_transformer_layer=2,
                          transformer_embed_dim=128,
                          transformer_input_pos_enc=False).cuda()
    pc = torch.randn(size=(4, 512, 3)).cuda()
    point_feats = torch.randn(size=(4, 512, 3)).cuda()
    mask = segnet(pc, point_feats)
    print (mask.shape)

    print('Number of parameters:', sum(p.numel() for p in segnet.parameters() if p.requires_grad))
    print('Number of parameters in PointNet++ encoder:', sum(p.numel() for p in segnet.SA_modules.parameters() if p.requires_grad))
    print('Number of parameters in PointNet++ decoder:', sum(p.numel() for p in segnet.FP_modules.parameters() if p.requires_grad))
    print('Number of parameters in MaskFormer head:', sum(p.numel() for p in segnet.MF_head.parameters() if p.requires_grad))

    print(segnet)