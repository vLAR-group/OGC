import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt


def accumulate_eval_results(segm, mask, ignore_npoint_thresh=0):
    """
    Accumulate evaluation results on a batch of samples
    :param segm: (B, N) torch tensor.
    :param mask: (B, N, K) torch tensor.
    :param ignore_npoint_thresh: threshold to ignore GT objects with too few points.
    :return:
        Pred_IoU: (N').
        Pred_Matched: (N').
        Confidence: (N').
        N_GT_Inst: An integer.
    """
    segm = segm.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = [], [], [], []
    n_batch = segm.shape[0]
    for b in range(n_batch):
        pred_iou, pred_matched, confidence, n_gt_inst = eval_segm(segm[b], mask[b], ignore_npoint_thresh=ignore_npoint_thresh)
        Pred_IoU.append(pred_iou)
        Pred_Matched.append(pred_matched)
        Confidence.append(confidence)
        N_GT_Inst.append(n_gt_inst)
    Pred_IoU = np.concatenate(Pred_IoU)
    Pred_Matched = np.concatenate(Pred_Matched)
    Confidence = np.concatenate(Confidence)
    N_GT_Inst = np.sum(N_GT_Inst)
    return Pred_IoU, Pred_Matched, Confidence, N_GT_Inst


def eval_segm(segm, mask, ignore_npoint_thresh=0):
    """
    :param segm: (N,).
    :param segm_pred: (N, K).
    :return:
        pred_iou: (N,).
        pred_matched: (N,).
        confidence: (N,).
        n_gt_inst: An integer.
    """
    segm_pred = np.argmax(mask, axis=1)
    _, segm, gt_sizes = np.unique(segm, return_inverse=True, return_counts=True)
    pred_ids, segm_pred, pred_sizes = np.unique(segm_pred, return_inverse=True, return_counts=True)
    n_gt_inst = gt_sizes.shape[0]
    n_pred_inst = pred_sizes.shape[0]
    mask = mask[:, pred_ids]

    # Compute Intersection
    intersection = np.zeros((n_gt_inst, n_pred_inst))
    for i in range(n_gt_inst):
        for j in range(n_pred_inst):
            intersection[i, j] = np.sum(np.logical_and(segm == i, segm_pred == j))

    # Ignore too small GT objects
    ignore_gt_ids = np.where(gt_sizes < ignore_npoint_thresh)[0]

    # An FP is not penalized, if mostly overlapped with ignored GT
    pred_ignore_ratio = np.sum(intersection[ignore_gt_ids], axis=0) / pred_sizes
    invalid_pred = (pred_ignore_ratio > 0.5)

    # Kick out predictions' area intersectioned with ignored GT
    pred_sizes = pred_sizes - np.sum(intersection[ignore_gt_ids], axis=0)
    valid_pred = np.logical_and(pred_sizes > 0, np.logical_not(invalid_pred))

    intersection = np.delete(intersection, ignore_gt_ids, axis=0)
    gt_sizes = np.delete(gt_sizes, ignore_gt_ids, axis=0)
    n_gt_inst = gt_sizes.shape[0]

    intersection = intersection[:, valid_pred]
    pred_sizes = pred_sizes[valid_pred]
    mask = mask[:, valid_pred]
    n_pred_inst = pred_sizes.shape[0]

    # Compute confidence scores for predictions
    confidence = np.zeros((n_pred_inst))
    for j in range(n_pred_inst):
        inst_mask = mask[segm_pred == j, j]
        confidence[j] = np.mean(inst_mask)

    # Find matched predictions
    union = np.expand_dims(gt_sizes, 1) + np.expand_dims(pred_sizes, 0) - intersection
    iou = intersection / union
    pred_iou = iou.max(axis=0)
    # In panoptic segmentation, Greedy gives the same result as Hungarian
    pred_matched = (pred_iou >= 0.5).astype(float)
    return pred_iou, pred_matched, confidence, n_gt_inst


"""
Average Precision (AP), Panoptic Quality (PQ), F1-score (F1), Precision (Pre) & Recall (Rec).
"""
def calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot=False, eps=1e-10):
    """
    AP computation with MS-COCO standards.
    :param Pred_Matched: (N).
    :param Confidence: (N).
    :param N_GT_Inst: An integer.
    """
    inds = np.argsort(-Confidence, kind='mergesort')
    Pred_Matched = Pred_Matched[inds]
    TP = np.cumsum(Pred_Matched)
    FP = np.cumsum(1 - Pred_Matched)
    precisions = TP / np.maximum(TP + FP, eps)
    recalls = TP / N_GT_Inst
    precisions, recalls = precisions.tolist(), recalls.tolist()

    for i in range(len(precisions) - 1, -0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Query 101-point
    recall_thresholds = np.linspace(0, 1, int(np.round((1 - 0) / 0.01)) + 1, endpoint=True)
    inds = np.searchsorted(recalls, recall_thresholds, side='left').tolist()
    precisions_queried = np.zeros(len(recall_thresholds))
    for rid, pid in enumerate(inds):
        if pid < len(precisions):
            precisions_queried[rid] = precisions[pid]
    precisions, recalls = precisions_queried.tolist(), recall_thresholds.tolist()
    AP = np.mean(precisions)
    
    # Plot P-R curve if needed
    if plot:
        fig = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        Pre, Rec = precisions[:2], recalls[:2]
        for i in range(1, len(precisions) - 1):
            Pre.extend([precisions[i+1], precisions[i+1]])
            Rec.extend([recalls[i], recalls[i+1]])
        Pre.append(precisions[-1])
        Rec.append(recalls[-1])

        plt.plot(Rec, Pre)
        plt.show()
        plt.close()
    return AP


def calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst, eps=1e-10):
    """
    :param Pred_IoU: (N).
    :param Pred_Matched: (N).
    :param N_GT_Inst: An integer.
    """
    TP = Pred_Matched.sum()
    TP_IoU = Pred_IoU[Pred_Matched > 0].sum()
    FP = Pred_Matched.shape[0] - TP
    FN = N_GT_Inst - TP

    PQ = TP_IoU / max(TP + 0.5*FP + 0.5*FN, eps)
    Pre = TP / max(TP + FP, eps)
    Rec = TP / max(TP + FN, eps)
    F1 = (2 * Pre * Rec) / max(Pre + Rec, eps)
    return PQ, F1, Pre, Rec


"""
Mean IoU (mIoU) & Rand Index (RI), adapted from MultiBodySync (CVPR'21).
"""
class ClusteringMetrics(nn.Module):
    # Mean IoU       (based on IoU Confusion Matrix)
    IOU = 1

    # Rand Index
    RI = 2

    def __init__(self, spec=None):
        super().__init__()
        if spec is None:
            self.spec = [self.IOU, self.RI]
        else:
            self.spec = spec

    def forward(self, mask, segm, ignore_npoint_thresh=0):
        """
        :param segm: (B, ...) torch.Tensor, starts from ID 0.
        :param mask: (B, ..., K) torch.Tensor.
        :param ignore_npoint_thresh: threshold to ignore GT objects with too few points.
        """
        output_dict = {}

        n_batch = mask.shape[0]
        gt_segm = segm.reshape(n_batch, -1).detach()                  # (B, N)
        n_data = gt_segm.shape[-1]
        n_gt_segms = gt_segm.max(dim=1).values + 1                    # (B,)

        k = mask.shape[-1]
        mask = mask.reshape(n_batch, -1, k).detach()  # (B, N, K)
        mask = mask.argmax(dim=-1)  # (B, N)

        k = max(k, n_gt_segms.max())
        mask = torch.eye(k, dtype=torch.float32,
                         device=mask.device)[mask]      # (B, N, K)
        gt_segm = torch.eye(k, dtype=torch.float32,
                            device=mask.device)[gt_segm]        # (B, N, K)

        # Ignore GT objects with too few points
        if ignore_npoint_thresh > 0:
            segm_size = gt_segm.sum(1, keepdim=True)
            nonsmall_mask = (segm_size >= ignore_npoint_thresh)
            nonsmall_segm = gt_segm * nonsmall_mask.float()
            valid_point = (nonsmall_segm.sum(-1) > 0)     # (B, N)
            invalid_point = torch.logical_not(valid_point)
            gt_segm[invalid_point] = 0
            mask[invalid_point] = 0
            valid_point = valid_point.float()
            valid_ri_mask = torch.bmm(valid_point.unsqueeze(2), valid_point.unsqueeze(1))     # (B, N, N)

        # Compute mIoU
        matching_score = torch.einsum('bng,bnp->bgp', gt_segm, mask)        # Intersection, (B, K, K)
        if self.IOU in self.spec:
            union_score = torch.sum(gt_segm, dim=1).unsqueeze(-1) + \
                          torch.sum(mask, dim=1, keepdim=True) - matching_score     # Union, (B, K, K)
            iou_score = matching_score / (union_score + 1e-8)       # IoU, (B, K, K)
            all_mean_ious = []
            for batch_id, n_gt_segm in enumerate(n_gt_segms):
                assert n_gt_segm <= k
                iou_confusion = iou_score[batch_id, :n_gt_segm, :].cpu().numpy()
                if ignore_npoint_thresh > 0:
                    nonsmall = nonsmall_mask[batch_id, 0, :n_gt_segm].cpu().numpy()
                    iou_confusion = iou_confusion[nonsmall]
                row_ind, col_ind = linear_sum_assignment(iou_confusion, maximize=True)
                current_iou = np.mean(iou_confusion[row_ind, col_ind])
                all_mean_ious.append(current_iou)
            output_dict["iou"] = all_mean_ious

        # Compute RI
        if self.RI in self.spec:
            ri_matrix_gt = torch.bmm(gt_segm, gt_segm.transpose(-1, -2))        # (B, N, N)
            ri_matrix_pd = torch.bmm(mask, mask.transpose(-1, -2))      # (B, N, N)
            if ignore_npoint_thresh > 0:
                ri = (valid_ri_mask * (ri_matrix_gt == ri_matrix_pd).float()).sum(dim=-1).sum(dim=-1) / valid_ri_mask.sum(dim=-1).sum(dim=-1)
            else:
                ri = torch.sum(ri_matrix_gt == ri_matrix_pd, dim=-1).sum(dim=-1).float() / (n_data * n_data)
            output_dict["ri"] = ri.cpu().numpy().tolist()
        return output_dict