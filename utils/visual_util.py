import numpy as np
import open3d as o3d


# Borrowed from PointGroup, with order adjusted
COLOR20 = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])


def build_pointcloud(pc, segm, with_background=False):
    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    if with_background:
        colors = np.concatenate((COLOR20[-1:], COLOR20[:-1]), axis=0)
    else:
        colors = COLOR20

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    if segm is not None:
        assert segm.shape[0] == pc.shape[0], f"Point and color id must have same size {segm.shape[0]}, {pc.shape[0]}"
        assert segm.ndim == 1, f"color id must be of size (N,) currently ndim = {segm.ndim}"
        point_cloud.colors = o3d.utility.Vector3dVector(colors[segm % colors.shape[0]] / 255.)
    return point_cloud


lines = [[0, 1], [1, 2], [2, 3], [0, 3],
         [4, 5], [5, 6], [6, 7], [4, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
box_colors = [[0, 1, 0] for _ in range(len(lines))]

def build_bbox3d(boxes):
    line_sets = []
    for corner_box in boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(box_colors)
        line_sets.append(line_set)
    return line_sets