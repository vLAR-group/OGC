# OGC: Unsupervised 3D Object Segmentation from Rigid Dynamics of Point Clouds

## Environment

Please first install a **GPU-supported pytorch** version which fits your machine.
We have tested with pytorch 1.9.0.

Install PointNet2 CPP lib:
```shell script
cd pointnet2
python setup.py install
cd ..
```

Install other dependencies:
```shell script
pip install -r requirements
```

**(Optional)** Install Open3D for the visualization of point cloud segmentation:
```shell script
pip install open3d
```

## Data preparation

### SAPIEN

Please download from links provided by [MultibodySync](https://github.com/huangjh-pub/multibody-sync):
- Train+Val (`mbs-shapepart`): [Google Drive](https://drive.google.com/file/d/1aGTn-PYxLjnhj9UKlv4YFV3Mt1E3ftci/view?usp=sharing)
- Test (`mbs-sapien`): [Google Drive](https://drive.google.com/file/d/1HR2X0DjgXLwp8K5n2nsvfGTcDMSckX5Z/view?usp=sharing)

Then put them into your `$SAPIEN` path.

### OGC-DR (Dynamic Room)

Please download the complete dataset from [???].

Alternatively, you can generate the dataset by yourself.
To do this, please first download the [ShapeNet Core v1](https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip).
Select the archives according to object categories specified in `data_prepare/ogcdr/meta.yaml` and unzip them into your `${OGC_DR}/ShapeNet_mesh` path.
Then run the following script to generate the dataset.
```shell script
python data_prepare/ogcdr/build_ogcdr.py $OGC_DR
```

### KITTI-SF (Scene Flow)

Please first download:
- [scene flow](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip) and [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip), from [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php)
- [semantic instance segmentation label](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip), from [KITTI Semantic Instance Segmentation Evaluation](http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015)

Merge the `training` folder of them in your `$KITTI_SF` path.
Then run the following script to unproject **disparity, optical flow, 2D segmentation** into **point cloud, scene flow, 3D segmentation**:
```shell script
python data_prepare/kittisf/process_kittisf.py $KITTI_SF
```

Finally, downsample all point clouds to 8192-point:
```shell script
python data_prepare/kittisf/downsample_kittisf.py $KITTI_SF --save_root ${KITTI_SF}_downsampled
# After extracting flow estimations in the following, come back here to downsample flow estimations
python data_prepare/kittisf/downsample_kittisf.py $KITTI_SF --save_root ${KITTI_SF}_downsampled --predflow_path flowstep3d
```
`${KITTI_SF}_downsampled` will be the path for the downsampled dataset.

### KITTI-Det (Detection)

Please first download the following items from [KITTI 3D Object Detection Evaluation 2017](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
- [velodyne point clouds](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
- [left color image](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
- [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- [object bounding box label](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)

Merge the `training` folder of them in your `$KITTI_DET` path.
Then run the following script to extract 8192-point **front-view** point cloud, and obtain segmentation from bounding box annotations.
```shell script
python data_prepare/kittidet/process_kittidet.py $KITTI_DET
```

## Pre-trained models

You can download all our pre-trained models from [here](https://www.dropbox.com/s/j2zgn04mlqaxf1w/OGC_ckpt.zip?dl=0) (including self-supervised scene flow networks, and unsupervised/supervised segmentation networks) and extract them to `./ckpt`.

## Scene flow estimation

### Train

Train the self-supervised scene flow networks:
```shell script
# SAPIEN dataset
python train_flow.py config/flow/sapien/sapien_unsup.yaml
# OGC-DR dataset
python train_flow.py config/flow/ogcdr/ogcdr_unsup.yaml
```
For KITTI-SF dataset, we directly employ the pre-trained model released by [FlowStep3D](https://github.com/yairkit/flowstep3d).

### Test

Evaluate and save the scene flow estimations.
```shell script
# SAPIEN dataset
python test_flow.py config/flow/sapien/sapien_unsup.yaml --split $SPLIT --save
# OGC-DR dataset
python test_flow.py config/flow/ogcdr/ogcdr_unsup.yaml --split $SPLIT --test_batch_size 12 --test_model_iters 5 --save
# KITTI-SF dataset
python test_flow_kittisf.py config/flow/kittisf/kittisf_unsup.yaml --split $SPLIT --test_model_iters 5 --save
```
`$SPLIT` can be train/val/test for SAPIEN & OGC-DR, train/val for KITTI-SF.

## Unsupervised segmentation

### Train

Alternate the segmentation network training and scene flow improvement. 
In each `$ROUND` (starting from 1):
```shell script
# SAPIEN dataset
python train_seg.py config/seg/sapien/sapien_unsup.yaml --round $ROUND
python oa_icp.py config/seg/sapien/sapien_unsup.yaml --split $SPLIT --round $ROUND --save

# OGC-DR dataset
python train_seg.py config/seg/ogcdr/ogcdr_unsup.yaml --round $ROUND
python oa_icp.py config/seg/ogcdr/ogcdr_unsup.yaml --split $SPLIT --round $ROUND --test_batch_size 24 --save

# KITTI-SF dataset
python train_seg.py config/seg/kittisf/kittisf_unsup.yaml --round $ROUND
python oa_icp.py config/seg/kittisf/kittisf_unsup.yaml --split $SPLIT --round $ROUND --test_batch_size 8 --save
```
When performing scene flow improvement, `$SPLIT` needs to traverse train/val/test for SAPIEN & OGC-DR, train/val for KITTI-SF.

### Test

```shell script
# SAPIEN dataset
python test_seg.py config/seg/sapien/sapien_unsup.yaml --split test --round $ROUND
# OGC-DR dataset
python test_seg.py config/seg/ogcdr/ogcdr_unsup.yaml --split test --round $ROUND --test_batch_size 16
# KITTI-SF dataset
python test_seg.py config/seg/kittisf/kittisf_unsup.yaml --split val --round $ROUND --test_batch_size 8
# KITTI-Det dataset
python test_seg.py config/seg/kittidet/kittisf_unsup.yaml --split val --round $ROUND --test_batch_size 8
```
`$ROUND` can be 1/2/3/..., and we take **2 rounds** as default in our experiments.
Specify `--save` to save the estimations. 
Specify `--visualize` for qualitative evaluation mode.

### Test the scene flow improvement

Your can follow the evaluation settings of FlowStep3D to test the improved flow, and see how our method push the boundaries of unsupervised scene flow estimation:
```shell script
python test_flow_kittisf_benchmark.py config/flow/kittisf/kittisf_unsup.yaml
```

## Supervised segmentation

You can train the segmentation network with full annotations.

### Train

```shell script
# SAPIEN dataset
python train_seg_sup.py config/seg/sapien/sapien_sup.yaml
# OGC-DR dataset 
python train_seg_sup.py config/seg/ogcdr/ogcdr_sup.yaml
# KITTI-SF dataset
python train_seg_sup.py config/seg/kittisf/kittisf_sup.yaml
# KITTI-Det dataset
python train_seg_sup.py config/seg/kittidet/kittidet_sup.yaml
```

### Test 

```shell script
# SAPIEN dataset
python test_seg.py config/seg/sapien/sapien_sup.yaml --split test
# OGC-DR dataset 
python test_seg.py config/seg/ogcdr/ogcdr_sup.yaml --split test --test_batch_size 16
# KITTI-SF dataset
python test_seg.py config/seg/kittisf/kittisf_sup.yaml --split val --test_batch_size 8
# KITTI-Det dataset
python test_seg.py config/seg/kittidet/kittisf_sup.yaml --split val --test_batch_size 8
python test_seg.py config/seg/kittidet/kittidet_sup.yaml --split val --test_batch_size 8
```

## Acknowledgements

Some code is borrowed from:
- [MultibodySync](https://github.com/huangjh-pub/multibody-sync) 
- [FlowStep3D](https://github.com/yairkit/flowstep3d)
- [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks)
- [KITTI Object data transformation and visualization](https://github.com/kuixu/kitti_object_vis)
- [ICP](https://github.com/ClayFlannigan/icp)