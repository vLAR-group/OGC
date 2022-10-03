import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from pointnet2.pointnet2 import furthest_point_sample, three_nn, three_interpolate


def fps_downsample(pc, n_sample_point=1024):
    """
    Downsample a point cloud with Furthest Point Sampling (FPS) and return indexes of sampled points.
    :param pc: (N, 3).
    :return:
        fps_idx: (N',).
    """
    pc = torch.from_numpy(pc).unsqueeze(0).cuda().contiguous()
    fps_idx = furthest_point_sample(pc, n_sample_point)
    fps_idx = fps_idx.cpu().numpy()[0]
    return fps_idx


def upsample_feat(pc, pc_fps, feat_fps):
    """
    Upsample per-point features from a downsampled point set to the complete point cloud.
    :param pc: (B, N, 3) torch.Tensor.
    :param pc_fps: (B, N', 3) torch.Tensor.
    :param feat_fps: (B, N', C) torch.Tensor.
    :return:
        flow: (B, N, C) torch.Tensor.
    """
    dist, nn_idx = three_nn(pc, pc_fps)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    feat_fps = feat_fps.transpose(1, 2).contiguous()
    feat = three_interpolate(feat_fps, nn_idx, weight)
    feat = feat.transpose(1, 2)
    return feat


def compress_label_id(segm):
    """
    Compress the object-id in segmentation to consecutive numbers starting from 0 (0, 1, 2, ...).
    :param segm: (N,).
    :return:
        segm_cpr: (N,).
    """
    _, segm_cpr = np.unique(segm, return_inverse=True)
    return segm_cpr


def segm_to_mask(segm, max_n_object=None):
    """
    Convert segmentation to one-hot mask.
    :param segm: (N,).
    """
    object_ids, segm_inv = np.unique(segm, return_inverse=True)
    if max_n_object is None:
        max_n_object = object_ids.shape[0]
    mask = np.eye(max_n_object, dtype=np.float32)[segm_inv]
    return mask


def batch_segm_to_mask(segms, max_n_object, ignore_npoint_thresh=0):
    """
    Convert a batch of segmentations to one-hot masks.
    :param segms: (B, N).
    :param max_n_object: an integer K.
    :return:
        masks: (B, N, K).
        valids: (B, N).
    """
    masks, valids = [], []
    for b in range(segms.shape[0]):
        segm = segms[b]
        # Ignore too small objects in GT
        if ignore_npoint_thresh > 0:
            object_ids, object_sizes = np.unique(segm, return_counts=True)
            object_ids_valid = object_ids[object_sizes >= ignore_npoint_thresh]
            valid = (np.in1d(segm, object_ids_valid))
            invalid = np.logical_not(valid)
            segm[invalid] = 0
        else:
            valid = np.ones_like(segm)
        valid = valid.astype(np.float32)
        valids.append(valid)

        # (N,) to (N, K)
        _, segm_inv = np.unique(segm, return_inverse=True)
        mask = np.eye(max_n_object, dtype=np.float32)[segm_inv]
        mask = mask * np.expand_dims(valid, 1)
        masks.append(mask)

    masks = np.stack(masks, 0)
    valids = np.stack(valids, 0)
    return masks, valids


def batch_segm_to_mask_withconf(segms, confs, max_n_object, ignore_npoint_thresh=0):
    """
    Convert a batch of segmentations to masks (objects have confidence scores, which will be kept in masks).
    :param segms: (B, N).
    :param confs: List [(K_b), ...]
    :param max_n_object: an integer K.
    :return:
        masks: (B, N, K).
        valids: (B, N).
    """
    masks, valids = [], []
    for b in range(segms.shape[0]):
        segm = segms[b]
        conf = confs[b]
        # Ignore too small objects in GT
        if ignore_npoint_thresh > 0:
            object_ids, object_sizes = np.unique(segm, return_counts=True)
            object_ids_valid = object_ids[object_sizes >= ignore_npoint_thresh]
            valid = (np.in1d(segm, object_ids_valid))
            invalid = np.logical_not(valid)
            segm[invalid] = 0
            conf = conf[object_ids_valid]
        else:
            valid = np.ones_like(segm)
        valid = valid.astype(np.float32)
        valids.append(valid)

        # (N,) to (N, K)
        _, segm_inv = np.unique(segm, return_inverse=True)
        mask = np.eye(max_n_object, dtype=np.float32)[segm_inv]
        mask = mask * np.expand_dims(valid, 1)
        # Add the confidence
        n_object = conf.shape[0]
        mask[:, :n_object] = mask[:, :n_object] * np.expand_dims(conf, 0)
        masks.append(mask)

    masks = np.stack(masks, 0)
    valids = np.stack(valids, 0)
    return masks, valids


def augment_transform(pcs, flows, aug_transform_args, n_view=2):
    """
    Augment the point cloud & flow with random spatial transformations.
    :param pcs: (2, N, 3).
    :param flows: (2, N, 3).
    :param aug_transform_args: a dict containing hyperparams for sampling spatial augmentations.
    :param n_view: number of transformations to be sampled.
    :return:
        aug_pcs: (V * 2, N, 3)
        aug_flows: (V * 2, N, 3)
    """
    assert pcs.shape[0] == flows.shape[0] == 2, 'Inconsistent number of frames!'
    pc1, pc2 = pcs[0], pcs[1]
    flow1, flow2 = flows[0], flows[1]

    aug_pcs, aug_flows = [], []
    for v in range(n_view):
        # Sample rotation R
        degree_range = np.array(aug_transform_args['degree_range'])
        degree = np.random.uniform(-degree_range, degree_range)
        rot = R.from_euler('zyx', degree, degrees=True).as_matrix()
        # Sample scaling s
        scale = np.random.uniform(aug_transform_args['scale_low'], aug_transform_args['scale_high'], 3)
        # Sample translation t
        shift_range = np.array(aug_transform_args['shift_range'])
        shift = np.random.uniform(-shift_range, shift_range)

        # Apply the transform: P' = sRP + t, F' = sRF
        aug_pc1 = scale * np.einsum('ij,nj->ni', rot, pc1) + shift
        aug_pc2 = scale * np.einsum('ij,nj->ni', rot, pc2) + shift
        aug_flow1 = scale * np.einsum('ij,nj->ni', rot, flow1)
        aug_flow2 = scale * np.einsum('ij,nj->ni', rot, flow2)

        # Augment the frame 2 separately, if needed (Only used for training scene flow estimation)
        if 'aug_pc2' in aug_transform_args.keys():
            aug_pc2_args = aug_transform_args['aug_pc2']
            # Sample rotation R
            degree_range = np.array(aug_pc2_args['degree_range'])
            degree = np.random.uniform(-degree_range, degree_range)
            rot2 = R.from_euler('zyx', degree, degrees=True).as_matrix()
            # Sample translation t
            shift_range = np.array(aug_pc2_args['shift_range'])
            shift2 = np.random.uniform(-shift_range, shift_range)

            # Apply the transform: aug_pc2, aug_flow2 & aug_flow1 change, aug_pc1 maintains
            aug_pc2_warped = aug_pc2 + aug_flow2
            aug_pc2 = np.einsum('ij,nj->ni', rot2, aug_pc2) + shift2
            aug_flow2 = aug_pc2_warped - aug_pc2
            aug_pc1_warped = aug_pc1 + aug_flow1
            aug_flow1 = np.einsum('ij,nj->ni', rot2, aug_pc1_warped) + shift2 - aug_pc1

        aug_pcs.extend([aug_pc1, aug_pc2])
        aug_flows.extend([aug_flow1, aug_flow2])

    aug_pcs, aug_flows = np.stack(aug_pcs, 0), np.stack(aug_flows, 0)
    return aug_pcs, aug_flows