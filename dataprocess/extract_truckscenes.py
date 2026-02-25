"""
# 
# Created: 2025-09-06 09:16
# Copyright (C) 2025-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang (https://kin-zhang.github.io/), Ajinkya Khoche (https://ajinkyakhoche.github.io/)
#
# Description: Preprocess Data, save as h5df format for faster loading
# This one is for MAN TruckScenes dataset
# 
# NOTE: truckscene follow really similar structure with TruckScenes format. That's why
# **NOT ALL** frames are annotated! So the SL training might be not that effective.
# Since truckscenes LiDAR is at 10Hz, we keep annotated flow to 10Hz also.
#
"""

from collections import defaultdict
import os
# os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import multiprocessing
from multiprocessing import Pool, current_process
from typing import Optional
from tqdm import tqdm
import numpy as np
import fire, time, h5py

from truckscenes import TruckScenes
from truckscenes.utils import splits
from truckscenes.utils.geometry_utils import transform_matrix
from truckscenes.utils.geometry_utils import points_in_box
from truckscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

import os, sys
PARENT_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..'))
sys.path.append(PARENT_DIR)
from dataprocess.misc_data import create_reading_index, check_h5py_file_exists, ManNamMap, find_closest_integer_in_ref_arr
from dataprocess.extract_nus import remove_ego_points, get_pose
from src.utils import npcal_pose0to1
from src.utils.av2_eval import CATEGORY_TO_INDEX
from linefit import ground_seg


GROUNDSEG_config = f"{PARENT_DIR}/conf/others/truckscenes.toml"
# NOTE(Qingwen): we only select 2x64 long-range LiDARs for processing, you can add more sensors if needed
# The timestamp reference will always be the last item in this list
SelectedSensor = ['LIDAR_RIGHT', 'LIDAR_LEFT'] # 2x64
# SelectedSensor = ['LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR'] # 4x32
# SelectedSensor = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR'] # all 6 LiDARs

def process_log(data_mode, data_dir: Path, scene_num_id: int, output_dir: Path, n: Optional[int] = None) :
    def create_group_data(group, pc, pose, lidar_id, lidar_center, gm = None, flow_0to1=None, flow_valid=None, flow_category=None, flow_instance=None, ego_motion=None):
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('pose', data=pose.astype(np.float64))
        group.create_dataset('ground_mask', data=gm.astype(bool))
        group.create_dataset('lidar_id', data=lidar_id.astype(np.float32))
        group.create_dataset('lidar_center', data=lidar_center.astype(np.float32)) # shape: [LiDAR_num, 3] (x, y, z)
        if ego_motion is not None:
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
        if flow_0to1 is not None:
            # ground truth flow information
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('flow_instance_id', data=flow_instance.astype(np.int16))

    def compute_flow_simple(data_fn, pc0, pose0, pose1, ts0, ts1, sample_ann_list, dclass, DataNameMap=ManNamMap):
        # compute delta transform between pose0 and pose1
        ego1_SE3_ego0 = npcal_pose0to1(pose0, pose1)
        # flow due to ego motion (baseline for all points)
        ego_flow = pc0[:,:3] @ ego1_SE3_ego0[:3,:3].T + ego1_SE3_ego0[:3,3] - pc0[:,:3]
        
        # object flow (without ego motion), used to track max flow magnitude
        obj_flow_all = np.zeros_like(pc0[:,:3])
        obj_flow_magnitude = np.zeros(len(pc0), dtype=np.float32)

        valid = np.ones(len(pc0), dtype=np.bool_)
        classes = np.zeros(len(pc0), dtype=np.uint8)
        instances = np.zeros(len(pc0), dtype=np.int16)

        delta_t = (ts1 - ts0) * 1e-6
        world_pc0 = pc0[:,:3] @ pose0[:3,:3].T + pose0[:3,3]
        id_ = 0
        for ann in sample_ann_list:
            # ann_vel = ann.velocity  # it's nan in truckscenes, we need to compute... don't know why
            ann_vel = data_fn.box_velocity(ann.token) # in world frame
            if np.isnan(ann_vel).any():
                continue

            # previous x/y/z velocity is on world frame, need to transform to ego frame
            ann_vel = ann_vel @ pose0[:3,:3]
            cls = ann.name

            # extend box length according to velocity, ref HiMo.
            ann.wlh[1] = ann.wlh[1] + (np.linalg.norm(ann_vel) * delta_t / 2)
            ann.wlh[2] = ann.wlh[2] + 0.2 # some truck top missing points, extend a bit
            points_in_box_mask = points_in_box(ann, world_pc0[:,:3].T, wlh_factor=1.1)
            classes[points_in_box_mask] = CATEGORY_TO_INDEX[DataNameMap[cls]]

            if np.sum(points_in_box_mask) > 5:
                obj_flow = ann_vel * delta_t
                obj_flow_mag = np.linalg.norm(obj_flow)
                
                # For overlapping boxes, keep the flow with higher magnitude
                higher_flow_mask = points_in_box_mask & (obj_flow_mag > obj_flow_magnitude)
                obj_flow_all[higher_flow_mask] = obj_flow
                obj_flow_magnitude[higher_flow_mask] = obj_flow_mag
                instances[higher_flow_mask] = (dclass[id_]+1)
                id_ += 1
            else:
                valid[points_in_box_mask] = False

        # Final flow = ego motion + object flow (with max magnitude selection)
        flow = ego_flow + obj_flow_all

        return {'flow_0_1': flow, 'valid_0': valid, 'classes_0': classes, 
                'ego_motion': ego1_SE3_ego0, 'flow_instance_id': instances}
    
    mants = TruckScenes(dataroot=data_dir, version=data_mode, verbose=False)
    scene = mants.scene[scene_num_id]
    log_id = scene['name']
    
    # In man-truckscenes, samples are annotated at 2 Hz and sweeps at 10 Hz. 
    sample_data_lst = []
    for sample in mants.sample:
        if sample['scene_token'] != scene['token']:
            continue
        else:
            sample_data_lst.append(sample)

    now_sample_token_str = scene['first_sample_token']
    sample = mants.get('sample', now_sample_token_str)

    # initialize full sweep data dict
    full_sweep_data_dict, timestamps = {}, {}
    for sensor_name in SelectedSensor:
        full_sweep_data_dict[sensor_name] = []
        timestamps[sensor_name] = []

    for channel, token in sample['data'].items():
        if channel in SelectedSensor:
            sample_data = mants.get('sample_data', token)
            while sample_data['next'] != '':
                ts0 = sample_data['timestamp']
                full_sweep_data_dict[channel].append(sample_data)
                timestamps[channel].append(ts0)
                sample_data = mants.get('sample_data', sample_data['next'])

    if check_h5py_file_exists(output_dir/f'{log_id}.h5', timestamps[SelectedSensor[-1]]):
        print(f'{log_id} already exists and all timestamps are , skip...')
        return            

    dclass = defaultdict(lambda: len(dclass))
    mygroundseg = ground_seg(GROUNDSEG_config)


    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, sweep_data in enumerate(full_sweep_data_dict[SelectedSensor[-1]]):
            ts0 = sweep_data['timestamp']
            pose0 = get_pose(mants, sweep_data, w2stf=False)

            lidar_list, lidar_center, lidar_dt, lidar_id = [], [], [], []; lidar_id_cnt = 0
            for single_sensor_name in full_sweep_data_dict.keys():
                # load closest sensor point cloud in reference ego frame
                sensor_sweep_list = full_sweep_data_dict[single_sensor_name]
                closest_ch_ind, closest_ch_timestamp, timestamp_diff = find_closest_integer_in_ref_arr(
                    ts0, np.array([t['timestamp'] for t in sensor_sweep_list])
                )
                sensor_sweep = sensor_sweep_list[closest_ch_ind]

                ego2lidar = mants.get('calibrated_sensor', sensor_sweep['calibrated_sensor_token'])
                ego2lidar_np = transform_matrix(ego2lidar['translation'], Quaternion(ego2lidar['rotation']))

                if 'LIDAR' in single_sensor_name:
                    pc = LidarPointCloud.from_file(os.path.join(str(data_dir), sensor_sweep['filename'])).points.T 
                    # NOTE(Qingwen): we need all points in base_link (ego) coordinate, but we save sensor center also
                    pc[:,:3] = pc[:,:3] @ ego2lidar_np[:3,:3].T + ego2lidar_np[:3,3].T
                    lidar_list.append(pc)
                    
                    # lidar_dt.append(np.ones(pc.shape[0]) * timestamp_diff * 1e-6) # microsecond to s
                    lidar_id.append(np.ones(pc.shape[0]) * lidar_id_cnt)
                    # lidar_center.append(ego2lidar_np[:3,3].T) # x, y, z
                    lidar_center.append(ego2lidar_np)
                    lidar_id_cnt += 1

            points = np.vstack(lidar_list)
            # lidar_dt = np.hstack(lidar_dt)
            lidar_id = np.hstack(lidar_id)
            lidar_center = np.array(lidar_center) # shape Num_LiDAR x 4x4

            points, not_close = remove_ego_points(points, length_threshold=2.0, width_threshold=7.0)
            lidar_id = lidar_id[not_close]
            # lidar_dt = lidar_dt[not_close]
            is_ground_0 = np.array(mygroundseg.run(points[:, :3]))

            # HARDCODE: for TruckScenes, add all points below 0.2m as ground
            is_ground_0 = is_ground_0 | (points[:,2] < 0.2)

            group = f.create_group(str(ts0))
            if cnt == len(full_sweep_data_dict[SelectedSensor[-1]]) - 1:
                create_group_data(group=group, pc=points, gm=is_ground_0.astype(np.bool_), pose=pose0, \
                                lidar_id=lidar_id, lidar_center=lidar_center)
            else:
                sweep_data_next = full_sweep_data_dict[SelectedSensor[-1]][cnt+1]
                ts1 = sweep_data_next['timestamp']
                pose1 = get_pose(mants, sweep_data_next, w2stf=False)
                
                # annotated frame, compute flow
                if sweep_data['is_key_frame'] and sweep_data['prev'] != "":
                    curr_scene_ann = mants.get_boxes(sweep_data['token'])
                    scene_flow = compute_flow_simple(mants, points, pose0, pose1, ts0, ts1, curr_scene_ann, dclass, DataNameMap=ManNamMap)

                    create_group_data(group=group, pc=points, gm=is_ground_0.astype(np.bool_), pose=pose0, \
                                    lidar_id=lidar_id, lidar_center=lidar_center, \
                                    flow_0to1=scene_flow['flow_0_1'], flow_valid=scene_flow['valid_0'], flow_category=scene_flow['classes_0'], \
                                    flow_instance=scene_flow['flow_instance_id'],
                                    ego_motion=scene_flow['ego_motion'])
                else: # no annotations, only save data
                    create_group_data(group=group, pc=points, gm=is_ground_0.astype(np.bool_), pose=pose0, \
                                        lidar_id=lidar_id, lidar_center=lidar_center, \
                                        ego_motion=npcal_pose0to1(pose0, pose1))

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(data_mode, data_dir: Path, scene_list: list, output_dir: Path, nproc: int):
    """
    Compute sceneflow for all logs in the dataset. 
    Logs are processed in parallel.
    """
    
    if not (data_dir).exists():
        print(f'{data_dir} not found')
        return

    args = sorted([(data_mode, data_dir, scene_num_id, output_dir) for scene_num_id in range(len(scene_list))])
    print(f'Using {nproc} processes')
    
    # # for debug
    # for x in tqdm(args[:2]):
    #     proc(x, ignore_current_process=True)
    #     # break

    if nproc <= 1:
        for x in tqdm(args):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(scene_list), ncols=100))

def main(
    data_dir: str = "/home/kin/data/man-demo/man-truckscenes",
    mode: str = "v1.0-mini",
    output_dir: str ="/home/kin/data/truckscenes/h5py",
    nproc: int = (multiprocessing.cpu_count() - 1),
    only_index: bool = False,
    split_name = None
):
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini'] # defined by nus.
    assert mode in available_vers
    # man = TruckScenes(dataroot=data_dir, version=mode, verbose=False)
    # print(f"Processing {mode} dataset with {len(man.scene)} scenes...")

    if mode == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
        if split_name is not None and split_name == 'train':
            input_dict = {'train': train_scenes}
        elif split_name is not None and split_name == 'val':
            input_dict = {'val': val_scenes}
        else:
            input_dict = {'train': train_scenes, 'val': val_scenes}
    elif mode == 'v1.0-test':
        test_scenes = splits.test
        input_dict = {'test': test_scenes}
    elif mode == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        input_dict = {
            'train': train_scenes,
            'val': val_scenes
        }
        # NOTE(Qingwen): or if you don't want to split mini, use below
        # input_dict = {'mini': train_scenes + val_scenes}
    else:
        raise ValueError('unknown')

    for input_key, input_val in input_dict.items():
        output_dir_ = Path(output_dir) / input_key
        print("[INFO] We are processing data to ", output_dir_)
        if only_index:
            create_reading_index(Path(output_dir_))
            create_reading_index(Path(output_dir_), flow_inside_check=True)
            return
        output_dir_.mkdir(exist_ok=True, parents=True)
        process_logs(mode, Path(data_dir), input_val, output_dir_, nproc)
        create_reading_index(output_dir_)
        create_reading_index(Path(output_dir_), flow_inside_check=True)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")