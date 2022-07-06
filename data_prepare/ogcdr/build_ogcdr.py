import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import tqdm
import argparse
import numpy as np
import trimesh
import random
import pickle
from scipy.spatial.transform import Rotation as R

from utils.data_util import fps_downsample


parser = argparse.ArgumentParser()
parser.add_argument('data_root', type=str, help='Root path for the dataset')
parser.add_argument('--keep_background', dest='keep_background', default=False, action='store_true', help='Keep the background in sampled point clouds or not')
args = parser.parse_args()

data_root = args.data_root
SRC_DIR = osp.join(data_root, 'ShapeNet_mesh')
SAVE_DIR = osp.join(data_root, 'data')
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_MESH_DIR = osp.join(data_root, 'mesh')
os.makedirs(SAVE_MESH_DIR, exist_ok=True)

# Fix random seed
np.random.seed(0)
random.seed(0)

# Object classes to use from ShapeNet
classes = ['02828884', '02933112', '03001627', '03211117', '03636649', '04256520', '04379243']

# Hyperparams for Objects
n_objects = [8, 7, 6, 5, 4]     # Number of objects per room (regarded as individual types)
scale_intervals = [
    [0.2, 0.3],
    [0.2, 0.35],
    [0.25, 0.35],
    [0.25, 0.40],
    [0.25, 0.45],
]       # Object scales for individual types
dataset_size = 1000     # Number of rooms per type
split_percentages = [.75, .05, .2]
split_names = ['train', 'val', 'test']
n_rooms = [int(p * dataset_size) for p in split_percentages]
y_angle_range = [0., 360.]

# Hyperparams for Background
xz_ground_range = [0.6, 1.]
ground_thickness = 0.01
ground_height = -0.5
ground_level = ground_height + ground_thickness
wall_thickness = 0.01
wall_height_range = [0.2, 0.4]

# Hyperparams for Dynamics
n_frame = 4
prob_rotation = 1.0
prob_rotation_y = 0.6
mot_y_angle_range = [-10., 10.]
mot_xz_angle_range = [-10., 10.]
mot_transl_range = [0.02, 0.04]  # Give a lower bound to enforce motion

# Hyperparams for sampling point cloud
n_sample_point = 100000
n_sample_point_fps = 2048


def get_class_models(cl, split_name):
    ''' Returns list of models for specific class and split.
    '''
    split_file = osp.join('data_prepare/ogcdr/raw_splits', cl, '%s.lst'%(split_name))
    with open(split_file, 'r') as f:
        model_files = f.read().split('\n')
        model_files = list(filter(lambda x: len(x) > 0, model_files))
    return model_files


def sample_models(model_dict, n_object):
    ''' Samples n_object from model_dict
    '''
    classes = [cl for cl in model_dict.keys()]
    classes.sort()

    out_list = []
    cl_list = []
    for n_object in range(n_object):
        cl = random.choice(classes)
        cl_list.append(cl)
        model = random.choice(model_dict[cl])
        out_list.append('%s/%s' % (cl, model))
    return out_list, cl_list


def sample_scales(n_object, type_id):
    '''Samples n_object scales in intervl scale_interval
    '''
    scale_interval = scale_intervals[type_id]
    out_list = [scale_interval[0] + np.random.rand() * (scale_interval[1] - scale_interval[0]) for i in range(n_object)]
    return out_list


def load_meshes(model_list, scale_list):
    """
    Loads the meshes in the list and scales according to provided list.
    The loaded meshes are with canonical poses (bounding box centered at origin).
    """
    out_list = []
    for model_idx, model in enumerate(model_list):
        cl, m = model.split('/')
        model_path = osp.join(SRC_DIR, cl, m, 'model.obj')
        mesh = trimesh.load(model_path, force='mesh')
        
        # Apply scaling
        bbox = mesh.bounds
        current_scale = (bbox[1] - bbox[0]).max()
        mesh.vertices = mesh.vertices / current_scale * scale_list[model_idx]
        
        # Center around origin
        mesh.vertices = mesh.vertices - bbox.sum(0) / 2
        out_list.append(mesh)
    return out_list


def sample_poses(mesh_list, y_angles):
    out_list = []
    pose_list = []
    for mesh_idx, mesh in enumerate(mesh_list):
        r = R.from_euler('y', [y_angles[mesh_idx]], degrees=True).as_matrix()
        mat = np.eye(4)
        mat[:3, :3] = r

        # Safe copy
        out_mesh = mesh.copy()
        out_mesh.apply_transform(mat)

        # Place mesh onto the ground
        y_transl = ground_level - min(out_mesh.vertices[:, 1])
        out_mesh.vertices[:, 1] = out_mesh.vertices[:, 1] + y_transl
        mat[1, 3] = y_transl

        out_list.append(out_mesh)
        pose_list.append(mat)
    return out_list, pose_list


def draw_sample(bounds, it=0, method='uniform', dx=0.1, sigma=0.05, xz_range=[1., 1.]):
    ''' Draws a sample for provided method and given bounding box.
    '''
    if method == 'uniform':
        loc0 = -xz_range / 2. + wall_thickness
        loc_len = xz_range - bounds - 2 * wall_thickness
        loc = loc0 + np.random.rand(2) * loc_len
    if method == 'gaussian':
        mu_list = [[-0.5 + dx, -0.5 + dx],
              [0.5 - dx, -0.5 + dx],
              [-0.5 + dx, 0.5 - dx],
              [0.5 - dx, 0.5 - dx],
              [0., 0.],
             ]
        while(True):
            loc = mu_list[it] + np.random.randn(2) * sigma
            if np.all(loc > -0.5) and np.all(loc + bounds < 0.5):
                break
    if method == 'uniform_structured':
        loc0 = [
            [-0.5, -0.5],
            [-0.5, 0.],
            [0., -0.5],
            [0., 0.],
        ]
        loc = loc0[it] + np.random.rand(2) * (0.5 - bounds)
    return loc


def check_intersection_interval(i1, i2):
    ''' Checks if the 2D intervals intersect.
    '''
    # i1, i2 of shape 2 x 2
    center_i1 = np.sum(i1, axis=0) / 2.
    center_i2 = np.sum(i2, axis=0) / 2.
    width_i1 = i1[1] - i1[0]
    width_i2 = i2[1] - i2[0]
    return np.all(abs(center_i1 - center_i2) < (width_i1 + width_i2) / 2)


def sample_locations(mesh_list, xz_range, poses, max_iter=1000):
    """
    Samples locations for the provided mesh list.
    """
    meshes = []
    bboxes = []
    poses_translated = []
    for mesh_idx, mesh in enumerate(mesh_list):
        # get bounds
        bounds = (mesh.bounds[1] - mesh.bounds[0])[[0, 2]]
        # sample location
        found_loc = False
        it = 0
        while(not found_loc):
            it += 1
            if it > max_iter:
                raise ValueError("Maximum number of iterations exceeded!")
            loc0 = draw_sample(bounds, method='uniform', it=mesh_idx, xz_range=xz_range)
            bbox_i = np.array([loc0, loc0 + bounds])
            found_loc = True
            for bbox in bboxes:
                if check_intersection_interval(bbox_i, bbox):
                    found_loc = False
                    break
        bboxes.append(bbox_i)

        # translate mesh
        xz_transl = loc0 - np.min(mesh.vertices[:, [0, 2]], axis=0)
        mesh.vertices[:, [0, 2]] = mesh.vertices[:, [0, 2]] + xz_transl
        meshes.append(mesh)
        # translate pose
        pose = poses[mesh_idx]
        pose[[0, 2], 3] = xz_transl
        poses_translated.append(pose)

    return meshes, poses_translated


def dynamic_poses(mesh_list, poses):
    out_list = []
    poses_rotated = []
    for mesh_idx, mesh in enumerate(mesh_list):
        pose = poses[mesh_idx]
        if np.random.rand() < prob_rotation:
            # Rotate around Y
            if np.random.rand() < prob_rotation_y:
                angle = mot_y_angle_range[0] + np.random.rand() * (mot_y_angle_range[1] - mot_y_angle_range[0])
                r = R.from_euler('y', [angle], degrees=True).as_matrix()
            # Rotate around X or Z
            else:
                angle = mot_xz_angle_range[0] + np.random.rand() * (mot_xz_angle_range[1] - mot_xz_angle_range[0])
                if np.random.rand() < 0.5:     # 0 for x-axis, 1 for z
                    r = R.from_euler('x', [angle], degrees=True).as_matrix()
                else:
                    r = R.from_euler('z', [angle], degrees=True).as_matrix()
            pose_rot = pose[:3, :3]
            pose_rot = np.matmul(r, pose_rot)
            mat = np.eye(4)
            mat[:3, :3] = pose_rot
            # Safe copy
            out_mesh = mesh.copy()
            out_mesh.apply_transform(mat)

            # Enforce object to lie on ground
            transl = pose[:3, 3]
            transl[1] = ground_level - min(out_mesh.vertices[:, 1])
            out_mesh.vertices = out_mesh.vertices + transl
            mat[:3, 3] = transl
        else:
            mat = pose.copy()
            out_mesh = mesh.copy()
            out_mesh.apply_transform(mat)

        out_list.append(out_mesh)
        poses_rotated.append(mat)

    return out_list, poses_rotated


def draw_dynamic_transl():
    transl = np.random.uniform(mot_transl_range[0], mot_transl_range[1], 2)
    pn = np.random.choice([True, False], 2)
    transl = np.where(pn, transl, -transl)
    return transl


def dynamic_locations(mesh_list, xz_range, poses, max_iter=1000):
    """
    Samples locations for the provided mesh list in the next frame.
    """
    meshes = []
    bboxes = []
    poses_translated = []
    for mesh_idx, mesh in enumerate(mesh_list):
        # get bounds
        bounds = (mesh.bounds[1] - mesh.bounds[0])[[0, 2]]
        last_loc0 = mesh.bounds[0][[0, 2]]
        loc0_lowerbound = -xz_range / 2. + wall_thickness
        loc0_upperbound = xz_range / 2. - wall_thickness - bounds

        # sample location
        found_loc = False
        it = 0
        while(not found_loc):
            it += 1
            if it > max_iter:
                raise ValueError("Maximum number of iterations exceeded!")
            xz_transl = draw_dynamic_transl()
            loc0 = last_loc0 + xz_transl
            if np.logical_or(loc0 < loc0_lowerbound, loc0 > loc0_upperbound).any():
                continue

            bbox_i = np.array([loc0, loc0 + bounds])
            found_loc = True
            for bbox in bboxes:
                if check_intersection_interval(bbox_i, bbox):
                    found_loc = False
                    break
        bboxes.append(bbox_i)

        # translate mesh
        mesh.vertices[:, [0, 2]] = mesh.vertices[:, [0, 2]] + xz_transl
        meshes.append(mesh)
        # translate pose
        pose = poses[mesh_idx]
        pose[[0, 2], 3] += xz_transl
        poses_translated.append(pose)

    return meshes, poses_translated


def sample_pointcloud(meshes, walls, ground, xz_range):
    """
    Sample point cloud from mesh list.
    """
    n_object = len(meshes)
    if args.keep_background:
        meshes = meshes + [ground] + walls
    
    c_vol = np.array([mesh.area for mesh in meshes])
    c_vol /= sum(c_vol)
    n_points = [int(c * n_sample_point) for c in c_vol]

    # Sample points from mesh surfaces
    points, segms = [], []
    for i, mesh in enumerate(meshes):
        pi, _ = trimesh.sample.sample_surface_even(mesh, n_points[i])
        if i < n_object:
            # Foreground object has segment ids starting from 1
            segm = (i+1) * np.ones(pi.shape[0], dtype=np.int16)
        else:
            segm = np.zeros(pi.shape[0], dtype=np.int16)
        points.append(pi)
        segms.append(segm)
    points = np.concatenate(points, axis=0).astype(np.float32)
    segms = np.concatenate(segms, axis=0).astype(np.int16)

    # Remove the thickness of ground & wall from pointcloud
    mask = points[:, 1] > (ground_level - 1e-4)
    mask &= points[:, 2] > (- xz_range[1] / 2. + wall_thickness - 1e-4)
    mask &= points[:, 0] > (- xz_range[0] / 2. + wall_thickness - 1e-4)
    mask &= points[:, 2] < (+ xz_range[1] / 2. - wall_thickness + 1e-4)
    mask &= points[:, 0] < (+ xz_range[0] / 2. - wall_thickness + 1e-4)
    points = points[mask]
    segms = segms[mask]

    # FPS downsample
    fps_idx = fps_downsample(points, n_sample_point=n_sample_point_fps)
    points = points[fps_idx]
    segms = segms[fps_idx]
    return points, segms


def get_y_angles(n_object):
    angles = y_angle_range[0] + np.random.rand(n_object) * (y_angle_range[1] - y_angle_range[0])
    return angles


def get_walls(xz_range=[1., 1.], wall_height=0.2):
    out_list = []

    wall_x = trimesh.creation.box((xz_range[0], wall_height, wall_thickness))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 2] = wall_x.vertices[:, 2] - min(wall_x.vertices[:, 2]) - xz_range[1] / 2.
    out_list.append(wall_x)

    wall_x = trimesh.creation.box((wall_thickness, wall_height, xz_range[1]))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 0] = wall_x.vertices[:, 0] - min(wall_x.vertices[:, 0]) - xz_range[0] / 2.
    out_list.append(wall_x)

    wall_x = trimesh.creation.box((xz_range[0], wall_height, wall_thickness))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 2] = wall_x.vertices[:, 2] - max(wall_x.vertices[:, 2]) + xz_range[1] / 2.
    out_list.append(wall_x)

    wall_x = trimesh.creation.box((wall_thickness, wall_height, xz_range[1]))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 0] = wall_x.vertices[:, 0] - max(wall_x.vertices[:, 0]) + xz_range[0] / 2.
    out_list.append(wall_x)
    return out_list


def get_ground(xz_range=[1., 1.]):
    x_len, z_len = xz_range
    ground = trimesh.creation.box((x_len, ground_thickness, z_len))
    bounds = ground.bounds
    ground.vertices = ground.vertices - (bounds.sum(0) / 2).reshape(1, 3) # center around origin
    ground.vertices[:, 1] = ground.vertices[:, 1] - min(ground.vertices[:, 1]) + ground_height
    return ground


# Main loop of data generation
split_lsts = {'train': '', 'val': '', 'test': ''}
for type_id, n_object in enumerate(n_objects):
    room_id = 0
    pbar = tqdm.tqdm(total=sum(n_rooms))

    for split_id, split_name in enumerate(split_names):
        # Load objects of all categories under current split (train/val/test)
        model_files = {}
        for cl in classes:
            model_files[cl] = get_class_models(cl, split_name)
        # Loop over items for current split
        split_item_id = 0

        while split_item_id < n_rooms[split_id]:
            # Create meta info for the scene
            item_dict = {}
            item_dict['room_id'] = room_id
            item_dict['split'] = split_name
            item_dict['n_object'] = n_object
            obj_list, cl_list = sample_models(model_files, n_object)
            item_dict['objects'] = obj_list
            item_dict['classes'] = cl_list
            item_dict['scales'] = sample_scales(n_object, type_id)
            axis0 = np.random.rand() > 0.5      # 0 is x-axis, 1 is z-axis
            scale_axis = np.random.rand() * (xz_ground_range[1] - xz_ground_range[0]) + xz_ground_range[0]
            ranges = [1., scale_axis] if axis0 else [scale_axis, 1.]
            item_dict['xz_ground_range'] = np.array(ranges)
            item_dict['wall_height'] = wall_height_range[0] + np.random.rand() * (wall_height_range[1] - wall_height_range[0])

            mesh_sequence, pose_sequence = [], []
            # Generate the 1st static frame
            canonical_meshes = load_meshes(item_dict['objects'], item_dict['scales'])
            init_y_angles = get_y_angles(n_object)
            meshes, poses = sample_poses(canonical_meshes, init_y_angles)
            try:
                meshes, poses = sample_locations(meshes, item_dict['xz_ground_range'], poses)
                mesh_sequence.append(meshes)
                pose_sequence.append(poses)
            except Exception as e:
                print('Error: ', e)
                continue

            # Generate dynamics
            frame_id = 1
            max_iter = 20
            it = 0
            while frame_id < n_frame:
                if it > max_iter:
                    break
                try:
                    meshes, poses = dynamic_poses(canonical_meshes, poses)
                    meshes, poses = dynamic_locations(meshes, item_dict['xz_ground_range'], poses)
                    frame_id += 1
                    mesh_sequence.append(meshes)
                    pose_sequence.append(poses)
                except Exception as e:
                    print('Error: ', e)
                    it += 1
                    continue
            # If not got enough frames after too many trials, sample a new scene
            if len(mesh_sequence) < n_frame:
                continue

            # Generate background
            walls = get_walls(xz_range=item_dict['xz_ground_range'], wall_height=item_dict['wall_height'])
            ground = get_ground(xz_range=item_dict['xz_ground_range'])

            # # Check if the pose matches the location
            # for frame_id in range(n_frame):
            #     meshes, poses = mesh_sequence[frame_id], pose_sequence[frame_id]
            #     for mesh, pose, mesh_c in zip(meshes, poses, canonical_meshes):
            #         mesh_t = mesh_c.copy().apply_transform(pose)
            #         print ((mesh_t.vertices - mesh.vertices).max(), (mesh_t.vertices - mesh.vertices).min())

            # Save path for the scene
            sample_name = '%02d_%06d'%(n_object, room_id)
            save_path = osp.join(SAVE_DIR, sample_name)
            os.makedirs(save_path, exist_ok=True)
            save_mesh_path = osp.join(SAVE_MESH_DIR, sample_name)
            os.makedirs(save_mesh_path, exist_ok=True)

            # Save meta info of the scene
            meta_file = osp.join(save_mesh_path, 'meta.pkl')
            with open(meta_file, 'wb') as f:
                pickle.dump(item_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save mesh models -- walls and ground (background)
            for i, wall in enumerate(walls):
                wall.export(osp.join(save_mesh_path, 'wall_%02d.obj'%(i)))
            ground.export(osp.join(save_mesh_path, 'ground.obj'))
            # Save mesh models -- objects (foreground)
            for frame_id in range(n_frame):
                meshes = mesh_sequence[frame_id]
                for mesh_idx, mesh in enumerate(meshes):
                    mesh.export(osp.join(save_mesh_path, 'object_%02d_%02d.obj'%(frame_id, mesh_idx)))

            # Save point cloud & segmentation & pose
            for frame_id in range(n_frame):
                meshes = mesh_sequence[frame_id]
                points, segms = sample_pointcloud(meshes, walls, ground, xz_range=item_dict['xz_ground_range'])
                np.save(osp.join(save_path, 'pc_%02d.npy'%(frame_id)), points)
                np.save(osp.join(save_path, 'segm_%02d.npy'%(frame_id)), segms)
                poses = pose_sequence[frame_id]
                poses = np.stack(poses)
                np.save(osp.join(save_path, 'pose_%02d.npy'%(frame_id)), poses)

            if split_lsts[split_name] != '':
                split_lsts[split_name] += '\n'
            split_lsts[split_name] += sample_name

            room_id += 1
            split_item_id += 1
            pbar.update(1)

    pbar.close()

# Save the split info
for split_name in split_names:
    with open(osp.join(SAVE_DIR, split_name + '.lst'), 'w') as f:
        f.write(split_lsts[split_name])