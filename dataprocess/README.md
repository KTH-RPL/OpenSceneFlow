Dataset
---

README for downloading and preprocessing the dataset. We includes waymo, argoverse 2.0 and nuscenes dataset in our project.

- [Download](#download): includes how to download the dataset.
- [Process](#process): run script to preprocess the dataset.

We've updated the process dataset for (Please cite the original dataset paper and involved work if you use them):

- [x] Argoverse 2.0: check [here](#argoverse-20). The process script Involved from [DeFlow](https://github.com/KTH-RPL/DeFlow).
- [x] Waymo: check [here](#waymo-dataset). The process script was involved from [SeFlow](https://github.com/KTH-RPL/SeFlow).
- [x] nuScenes: check [here](#nuscenes), The process script was involved from [DeltaFlow](https://github.com/Kin-Zhang/DeltaFlow).
- [x] ZOD (w/o gt): check [here](#zod-dataset). The process script was involved from [HiMo](https://kin-zhang.github.io/HiMo). (It could be a good first reference for users to extract other datasets in the future.)
- [x] TruckScene: check [here](#truckscene). The process script was involved from [DoGFlow](https://github.com/ajinkyakhoche/DoGFlow).

If you want to **use all datasets above**, there is a **specific environment** in [envsftool.yaml](../envsftool.yaml) to install all the necessary packages. As Waymo package have different configuration and conflict with the main environment. Setup through the following command:

```bash
conda env create -f envsftool.yaml
conda activate sftool
# NOTE we need **manually reinstall numpy** (higher than 1.22)
# * since waymo package force numpy==1.21.5, BUT!
# * hdbscan w. numpy<1.22.0 will raise error: 'numpy.float64' object cannot be interpreted as an integer
# * av2 need numpy >=1.22.0, waymo with numpy==1.22.0 will be fine on code running.
pip install numpy==1.22
```

## Download

### Argoverse 2.0

Install their download tool `s5cmd`, already in our envsftool.yaml. Then download the dataset:
```bash
# train is really big (700): totally 966 GB
s5cmd --numworkers 12 --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*" av2/sensor/train 

# val (150) and test (150): totally 168GB + 168GB
s5cmd --numworkers 12 --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*" av2/sensor/val
s5cmd --numworkers 12 --no-sign-request cp "s3://argoverse/datasets/av2/sensor/test/*" av2/sensor/test

# for local and online eval mask from official repo
s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .
```

Then to quickly pre-process the data, we can [read these commands](#process) on how to generate the pre-processed data for training and evaluation. This will take around 0.5-2 hour for the whole dataset (train & val) based on how powerful your CPU is.

Optional: More [self-supervised data in AV2 LiDAR only](https://www.argoverse.org/av2.html#lidar-link), note: It **does not** include **imagery or 3D annotations**. The dataset is designed to support research into self-supervised learning in the lidar domain, as well as point cloud forecasting. 
```bash
# train is really big (16000): totally 4 TB
s5cmd --numworkers 12 --no-sign-request cp "s3://argoverse/datasets/av2/lidar/train/*" av2/lidar/train

# val (2000): totally 0.5 TB
s5cmd --numworkers 12 --no-sign-request cp "s3://argoverse/datasets/av2/lidar/val/*" av2/lidar/val

# test (2000): totally 0.5 TB
s5cmd --numworkers 12 --no-sign-request cp "s3://argoverse/datasets/av2/lidar/test/*" av2/lidar/test
``` 

#### Dataset frames

<!-- Note that some frames in LiDAR don't have any point cloud.... we didn't remove them in the total num. -->

| Dataset      | # Total Scene | # Total Frames |
| ------------ | ------------- | -------------- |
| Sensor/train | 700           | 110071         |
| Sensor/val   | 150           | 23547          |
| Sensor/test  | 150           | 23574          |
| LiDAR/train  | 16000         | -              |
| LiDAR/val    | 2000          | 597590         |
| LiDAR/test   | 2000          | 597575         |

### nuScenes

You need sign up an account at [nuScenes](https://www.nuscenes.org/) to download the dataset from [https://www.nuscenes.org/nuscenes#download](https://www.nuscenes.org/nuscenes#download) Full dataset (v1.0), you can choose to download lidar only. Click donwload mini split and unzip the file to the `nuscenes` folder if you want to test.

![](../assets/docs/nuscenes.png)


Extracting & processing nuScenes require special handling:

* Frame Rate: The raw LiDAR data is captured at 20Hz, while ground truth (GT) annotations are only available at 2Hz.
* Resampling: To standardize the data for consistent evaluation, we downsample the LiDAR point clouds to 10Hz. It is a GT-guided process that guarantees all annotated 2Hz frames are preserved within the final 10Hz sequence.
* The ground truth scene flow is generated using the official per-object velocity labels provided in the dataset, calculated between the resampled 10Hz frames.


#### Dataset frames

| Dataset | # Total Scene | # Total Frames |
| ------- | ------------- | -------------- |
| train   | 700           | 137575 / 27392 (w. gt)         |
| val     | 150           | 29126 / 5798 (w.gt)          |

### Waymo Dataset

To download the Waymo dataset, you need to register an account at [Waymo Open Dataset](https://waymo.com/open/). You also need to install gcloud SDK and authenticate your account. Please refer to [this page](https://cloud.google.com/sdk/docs/install) for more details. 

For cluster without root user, check [here sdk tar gz](https://cloud.google.com/sdk/docs/downloads-versioned-archives).

Website to check their file: [https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow)

The thing we need is all things about `lidar`, to download the data, you can use the following command:

```bash
gsutil -m cp -r "gs://waymo_open_dataset_scene_flow/valid" .
gsutil -m cp -r "gs://waymo_open_dataset_scene_flow/train" .
```

For ground segmentation, we follow the same style of [ZeroFlow](https://github.com/kylevedder/zeroflow/blob/master/data_prep_scripts/waymo/extract_flow_and_remove_ground.py) to have HDMap. You can download the processed map by running:

```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/waymo_map.tar.gz
tar -xvf waymo_map.tar.gz -C /home/kin/data/waymo/flowlabel
# you will see there is a `map` folder in the `flowlabel` folder now.
```

<!-- Another way to have ground mask is to use [linefit](https://github.com/Kin-Zhang/linefit) to generate the ground mask without effort. -->

#### Dataset frames

| Dataset | # Total Scene | # Total Frames |
| ------- | ------------- | -------------- |
| train   | 799           | 155687         |
| val     | 203           | 39381          |

### ZOD Dataset

Although ZOD have the most dense LiDAR sensor (128-channel), the dataset itself **does not include ground truth flow**. 
We provide the extraction script for Self-Supervised Learning (SSL) to train and visualize the results etc, again **no evaluation available** here.

To download the ZOD dataset, you need follow [the instruction here](https://zod.zenseact.com/download/): send email and ask for the download from the team.

For HiMo, we only downloaded [drives-set](https://zod.zenseact.com/drives/) for test purpose etc. The total drives-set includes 29 sequences (Total size: 303G). Here are [quick video play](https://www.bilibili.com/video/BV1Sh4y1z7v2) for each scene in the drives.

Please check the scripts: [dataprocess/extract_zod.py](./extract_zod.py) in detail, current we only process one scene while feel free to comment out for all scene etc.

### TruckScene

Please visit the [TruckScene dataset](https://brandportal.man/d/QSf8mPdU5Hgj/downloads#/-/dataset) page for privacy policy. You can download the dataset by following command:

```bash
# mini set, ~11G recommended for debugging purpose
cd /home/kin/data/truckscene/mini
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/mini/man-truckscenes_metadata_v1.0-mini.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/mini/man-truckscenes_sensordata_v1.0-mini.zip
unzip "man-truckscenes_*.zip"

# full trainval set, ~630G
cd /home/kin/data/truckscene/trainval
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_metadata_v1.0-trainval.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_sensordata01_v1.0-trainval.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_sensordata02_v1.0-trainval.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_sensordata03_v1.0-trainval.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_sensordata04_v1.0-trainval.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_sensordata05_v1.0-trainval.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_sensordata06_v1.0-trainval.zip
wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/trainval/man-truckscenes_sensordata07_v1.0-trainval.zip
unzip "man-truckscenes_*.zip"
```

Folder structure:
```
truckscene/
  — samples/
  — sweeps/
  — v1.0-mini/
  — v1.0-trainval/
```

#### Dataset frames

| Dataset | # Total Scene | # Total Frames |
| ------- | ------------- | -------------- |
| train   | 524           | 101902 / 20380 (w. gt)         |
| val     | 75           | 14625 / 2925 (w. gt)          |



## Process

This directory contains the scripts to preprocess the datasets into `.h5` files. 

- `extract_av2.py`: Process the datasets in Argoverse 2.0.
- `extract_nus.py`: Process the datasets in nuScenes.
- `extract_waymo.py`: Process the datasets in Waymo.
- `extract_zod.py`: Process the datasets in ZOD.
- `extract_truckscene.py`: Process the datasets in TruckScene.

Example Running command, you can also check our [slurm data-process script](../assets/slurm/data-process.sh) for more details.:
```bash
# av2:
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/h5py

# waymo:
python dataprocess/extract_waymo.py --mode train --flow_data_dir /home/kin/data/waymo/flowlabel --map_dir /home/kin/data/waymo/flowlabel/map --output_dir /home/kin/data/waymo/h5py  --nproc 48

# nus:
python dataprocess/extract_nus.py --mode v1.0-trainval --output_dir /home/kin/data/nus/h5py/full --nproc 24

# truckscene:
python dataprocess/extract_truckscene.py --data_dir /home/kin/data/man-truckscenes --mode v1.0-mini --output_dir /home/kin/data/nus/h5py
```


Extract all Argoverse 2.0 data to unified `.h5` format.
[Runtime: Normally need 45 mins finished run following commands totally in setup mentioned in our paper]
```bash
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/h5py
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --mask_dir /home/kin/data/av2/3d_scene_flow
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --mask_dir /home/kin/data/av2/3d_scene_flow
```

All these preprocess scripts will generate the same format `.h5` file. The file contains the following in codes:

File: `[*:logid].h5` file named in logid. Every timestamp is the key of group (f[key]).

```python
def process_log(data_dir: Path, log_id: str, output_dir: Path, n: Optional[int] = None) :
    def create_group_data(group, pc, gm, pose, flow_0to1=None, flow_valid=None, flow_category=None, ego_motion=None):
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('ground_mask', data=gm.astype(bool))
        group.create_dataset('pose', data=pose.astype(np.float32))
        if flow_0to1 is not None:
            # ground truth flow information
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
```

After preprocessing, all data can use the same dataloader to load the data. As already in our DeFlow code.

Or you can run testing file to visualize the data. 

```bash
# view gt flow
python tools/visualization.py --data_dir /home/kin/data/av2/h5py/sensor/mini --res_name flow

python tools/visualization.py --data_dir /home/kin/data/waymo/h5py/val --res_name flow
```

### Self-Supervised Process

Process train data for self-supervised learning. Only training data needs this step. 
[Runtime: Normally need 15 hours for my desktop, 3 hours for the cluster with five available nodes parallel running.]

```bash
python process.py --data_dir /home/kin/data/av2/h5py/sensor/train --scene_range 0,701
```

As some users must have multi-nodes for running, here I provide an example SLURM script to run the data process in parallel. 
Check [assets/slurm/ssl-process.sh](../assets/slurm/ssl-process.sh) for more details.