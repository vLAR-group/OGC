import os
import os.path as osp
import argparse
from tqdm import tqdm

import open3d as o3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str,
                        default='/home/ziyang/Desktop/Datasets/OGC_DynamicRoom', help='Root path for the OGC-DR dataset')
    parser.add_argument('--dest_root', type=str,
                        default='/home/ziyang/Desktop/Datasets/OGC_DynamicRoom_SingleView', help='Root path for the OGC-DRSV dataset')
    args = parser.parse_args()

    data_root = osp.join(args.src_root, 'mesh')
    data_ids = sorted(os.listdir(data_root))
    n_frame = 4
    save_root = osp.join(args.dest_root, 'pcd')
    os.makedirs(save_root, exist_ok=True)


    # Iterate over the dataset
    pbar = tqdm(total=len(data_ids))
    for data_id in data_ids:
        n_object = int(data_id[:2])
        data_path = osp.join(data_root, data_id)
        save_path = osp.join(save_root, data_id)
        os.makedirs(save_path, exist_ok=True)

        for frame_id in range(n_frame):
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible = False)

            # Load the objects and merge
            pcds = []
            for obj_id in range(n_object):
                pcd = o3d.io.read_triangle_mesh(osp.join(data_path, 'object_%02d_%02d.obj'%(frame_id, obj_id)))
                pcds.append(pcd)
            pcd = pcds[0]
            for i in range(1, len(pcds)):
                pcd += pcds[i]
            vis.add_geometry(pcd)

            # ctrl = vis.get_view_control()
            # ctrl.change_field_of_view(step=30)
            # vis.run()

            vis.poll_events()
            vis.update_renderer()
            save_file = osp.join(save_path, 'pc_%02d.pcd'%(frame_id))
            vis.capture_depth_point_cloud(save_file, convert_to_world_coordinate=True)
            vis.destroy_window()

        pbar.update(1)

    pbar.close()