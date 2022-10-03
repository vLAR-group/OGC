import torch


def eval_flow(gt_flow, flow_pred, epe_norm_thresh=0.05, eps=1e-10):
    """
    Compute scene flow estimation metrics: EPE3D, Acc3DS, Acc3DR, Outliers3D.
    :param gt_flow: (B, N, 3) torch.Tensor.
    :param flow_pred: (B, N, 3) torch.Tensor.
    :param epe_norm_thresh: Threshold for abstract EPE3D values, used in computing Acc3DS / Acc3DR / Outliers3D and adapted to sizes of different datasets.
    :return:
        epe & acc_strict & acc_relax & outlier: Floats.
    """
    gt_flow = gt_flow.detach().cpu()
    flow_pred = flow_pred.detach().cpu()

    epe_norm = torch.norm(flow_pred - gt_flow, dim=2)
    sf_norm = torch.norm(gt_flow, dim=2)
    relative_err = epe_norm / (sf_norm + eps)
    epe = epe_norm.mean().item()

    # Adjust the threshold to the scale of dataset
    acc_strict = (torch.logical_or(epe_norm < epe_norm_thresh, relative_err < 0.05)).float().mean().item()
    acc_relax = (torch.logical_or(epe_norm < (2 * epe_norm_thresh), relative_err < 0.1)).float().mean().item()
    outlier = (torch.logical_or(epe_norm > (6 * epe_norm_thresh), relative_err > 0.1)).float().mean().item()
    return epe, acc_strict, acc_relax, outlier