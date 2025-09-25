"""
# 
# Created: 2024-02-24 10:48
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen ZHANG  (https://kin-zhang.github.io/), Ajinkya Khoche, Peizheng Li
#
# Description: Preprocess Data, save as h5df format for faster loading
# This one is for nuScenes dataset
#
# NOTE: nuscenes dataset need resample to 10Hz to align all other evaluations (e.g metric and frequency).
# and NOT ALL frames are annotated! So the SL training might be not that effective.
#
"""

import os
# os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional
from tqdm import tqdm
import numpy as np
import fire, time, h5py
from collections import defaultdict

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

import os, sys
PARENT_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..'))
sys.path.append(PARENT_DIR)
from dataprocess.misc_data import create_reading_index, check_h5py_file_exists, NusNamMap
from src.utils import npcal_pose0to1
from src.utils.av2_eval import CATEGORY_TO_INDEX
from linefit import ground_seg

GROUNDSEG_config = f"{PARENT_DIR}/conf/others/nuscenes.toml"

def remove_ego_points(pc: np.ndarray,
                    length_threshold: float = 4.084 / 2, 
                    width_threshold: float = 1.730 / 2) -> np.ndarray:
    """
    Remove ego points from a point cloud.
    :param pc: point cloud to remove ego points from, shape: (N, 4)
    :param length_threshold: Length threshold.
    :param width_threshold: Width threshold.
    :return: point cloud without ego points, shape: (N, 4).
    """
    # NuScenes LiDAR position
    # x: left --> width_threshold; y: front --> length_threshold; z: up --> height_threshold
    x_filt = np.logical_and(pc[:, 0] > -width_threshold, pc[:, 0] < width_threshold)
    y_filt = np.logical_and(pc[:, 1] > -length_threshold, pc[:, 1] < length_threshold)
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    not_close = not_close.astype(bool)
    return pc[not_close], not_close

def _load_points_from_file(filename: str) -> np.ndarray:
    """
    Private function to load point cloud from file.
    :param filename: Path of the point cloud file.
    :return: Point cloud as numpy array.
    """
    pc = np.fromfile(filename, dtype=np.float32)
    pc = pc.reshape((-1, 5))[:, :4]
    return pc
        
def get_pose(data_fn, sweep_data, w2stf=True):
    world2ego = data_fn.get('ego_pose', sweep_data['ego_pose_token'])
    # without considering the sensor to ego transform, we will do it outside if multiple sensors are used.
    if not w2stf:
        return transform_matrix(world2ego['translation'], Quaternion(world2ego['rotation'])).astype(np.float64)

    ego2lidar = data_fn.get('calibrated_sensor', sweep_data['calibrated_sensor_token'])
    ego2lidar_np = transform_matrix(ego2lidar['translation'], Quaternion(ego2lidar['rotation']))
    world2ego_np = transform_matrix(world2ego['translation'], Quaternion(world2ego['rotation']))
    return np.dot(world2ego_np, ego2lidar_np)
    
def if_annotated_frame(sample_ann_dict, ts0):
    gt_flow_flag = False
    for key, anno_value in sample_ann_dict.items():
        if ts0 in anno_value.keys():
            gt_flow_flag = True # if there is one anno have gt vel then set to compute flow.
            break
    return gt_flow_flag

def _resample_data(nusc, sample_data, sample_ann_dict, datafrequency=20, resample2frequency=10):
    """
    NOTE(Qingwen) - 2025-05-18:
    We always want to start from the first GT frame, and then resample the data! So we have as many GT frames as possible...
    """
    sweep_data_lst, timestamps_lst = [], []
    skipFrame = int(datafrequency / resample2frequency)  # since nuscenes sweep at 20Hz, we want to resample to 10Hz
    cnt = 0

    # Find the first GT frame
    while sample_data['next'] != '':
        ts0 = sample_data['timestamp']
        gt_flow_flag = if_annotated_frame(sample_ann_dict, ts0)
        
        if gt_flow_flag:
            # Found first GT frame, add it and break
            sweep_data_lst.append(sample_data)
            timestamps_lst.append(ts0)
            sample_data = nusc.get('sample_data', sample_data['next'])
            cnt = 1  # Reset counter after finding first GT
            break
        
        sample_data = nusc.get('sample_data', sample_data['next'])

    # Continue processing from first GT frame
    while sample_data['next'] != '':
        ts0 = sample_data['timestamp']
        gt_flow_flag = if_annotated_frame(sample_ann_dict, ts0)
        
        # Always include GT frames
        if gt_flow_flag:
            sweep_data_lst.append(sample_data)
            timestamps_lst.append(ts0)
            sample_data = nusc.get('sample_data', sample_data['next'])
            cnt = 1  # Reset counter after each GT frame
        elif cnt % skipFrame == 0:
            # Include non-GT frames according to sampling rate
            sweep_data_lst.append(sample_data)
            timestamps_lst.append(ts0)
            sample_data = nusc.get('sample_data', sample_data['next'])
            cnt += 1
        else:
            # Skip this frame
            sample_data = nusc.get('sample_data', sample_data['next'])
            cnt += 1
    return sweep_data_lst, timestamps_lst

def process_log(nusc_mode, data_dir: Path, scene_num_id: int, output_dir: Path, resample2frequency=10, n: Optional[int] = None) :

    def create_group_data(group, pc, pose, gm = None, flow_0to1=None, flow_valid=None, flow_category=None, flow_instance=None, ego_motion=None):
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('pose', data=pose.astype(np.float64))
        group.create_dataset('ground_mask', data=gm.astype(bool))
        if ego_motion is not None:
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
        if flow_0to1 is not None:
            # ground truth flow information
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('flow_instance_id', data=flow_instance.astype(np.int16))

    def compute_flow_simple(pc0, pose0, pose1, ts0, ts1, sample_ann_dict, dclass):
        # compute delta transform between pose0 and pose1
        ego1_SE3_ego0 = npcal_pose0to1(pose0, pose1)
        # flow due to ego motion
        flow = np.zeros_like(pc0[:,:3])
        flow = pc0[:,:3] @ ego1_SE3_ego0[:3,:3].T + ego1_SE3_ego0[:3,3] - pc0[:,:3] # pose flow
        
        valid = np.ones(len(pc0), dtype=np.bool_)
        classes = np.zeros(len(pc0), dtype=np.uint8)
        instances = np.zeros(len(pc0), dtype=np.int16)

        delta_t = (ts1 - ts0) * 1e-6
        world_pc0 = pc0[:,:3] @ pose0[:3,:3].T + pose0[:3,3]
        id_ = 0
        for key, anno_value in sample_ann_dict.items():
            # return: <np.float: 3>. Velocity in x/y/z direction in m/s.
            if ts0 not in anno_value.keys():
                continue
            ann_vel = anno_value[ts0]['vel']
            if np.isnan(ann_vel).any():
                continue
            # previous x/y/z velocity is on world frame, need to transform to ego frame
            ann_vel = ann_vel @ pose0[:3,:3]
            box0 = anno_value[ts0]['bbx']
            cls = anno_value[ts0]['class']

            # FIXME: compute points_in_box mask, expansion factor 1.1 here.
            points_in_box_mask = points_in_box(box0, world_pc0[:,:3].T, wlh_factor=1.1)
            classes[points_in_box_mask] = CATEGORY_TO_INDEX[NusNamMap[cls]]
            if np.sum(points_in_box_mask) > 5:
                obj_flow = np.ones_like(pc0[points_in_box_mask,:3]) * ann_vel * delta_t
                flow[points_in_box_mask] += obj_flow
                instances[points_in_box_mask] = (dclass[id_]+1)
                id_ += 1
            else:
                valid[points_in_box_mask] = False

        return {'flow_0_1': flow,
                'valid_0': valid, 'classes_0': classes, 
                'ego_motion': ego1_SE3_ego0,
                'flow_instance_id': instances}
     
    nusc = NuScenes(dataroot=data_dir, version=nusc_mode, verbose=False)
    scene = nusc.scene[scene_num_id]
    log_id = scene['name']

    # In nuscenes, samples are annotated at 2 Hz and sweeps at 20 Hz. 
    sample_data_lst = []
    for sample in nusc.sample:
        if sample['scene_token'] != scene['token']:
            continue
        else:
            sample_data_lst.append(sample)
    sample_ann_dict = dict()

    # step: create a dictionary, with keys denoting the annotation token,  
    # and value denoting a list with annotation value and timestamp
    for sample in sample_data_lst:
        for ann_token in sample['anns']:
            annotation = nusc.get('sample_annotation', ann_token)
            if annotation['instance_token'] not in sample_ann_dict.keys():
                sample_ann_dict[annotation['instance_token']] = {}
            # each annotation is list with attr 
            sample_ann_dict[annotation['instance_token']][sample['timestamp']]= {
                'bbx': Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation'])), \
                'vel': nusc.box_velocity(ann_token).tolist(), \
                'class': annotation['category_name']}
                
    # step note down timestamps for sweeps (interpolation points) and samples (at which data for interpolation exists)
    now_sample_token_str = scene['first_sample_token']
    sample = nusc.get('sample', now_sample_token_str)
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sweep_data_lst, timestamps = _resample_data(nusc, sample_data, sample_ann_dict, resample2frequency)

    if check_h5py_file_exists(output_dir/f'{log_id}.h5', timestamps):
        print(f'{log_id} already exists and all timestamps are , skip...')
        return
    
    dclass = defaultdict(lambda: len(dclass))
    mygroundseg = ground_seg(GROUNDSEG_config)
    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, sweep_data in enumerate(sweep_data_lst):
            ts0 = sweep_data['timestamp']
            # lidar point cloud
            pc0 = _load_points_from_file(os.path.join(nusc.dataroot, sweep_data['filename']))
            pc0, not_close = remove_ego_points(pc0)
            if pc0.shape[0] < 10:
                print(f'{log_id}/{ts0} has no points....')
                continue
            is_ground_0 = mygroundseg.run(pc0[:, :3])
            pose0 = get_pose(nusc, sweep_data)

            if cnt == len(sweep_data_lst) - 1:
                group = f.create_group(str(ts0))
                create_group_data(group=group, pc=pc0, gm=is_ground_0.astype(np.bool_), pose=pose0.astype(np.float32))
            else:
                sweep_data_next = sweep_data_lst[cnt+1] 
                ts1 = sweep_data_next['timestamp']
                pose1 = get_pose(nusc, sweep_data_next)
                gt_flow_flag = if_annotated_frame(sample_ann_dict, ts0)

                group = f.create_group(str(ts0))
                if not gt_flow_flag: # no annotations, only save data
                    create_group_data(group=group, pc=pc0, gm=is_ground_0.astype(np.bool_), pose=pose0.astype(np.float32), \
                                        ego_motion=(np.linalg.inv(pose1) @ pose0).astype(np.float32))
                else: # compute sceneflow
                    scene_flow = compute_flow_simple(pc0, pose0, pose1, ts0, ts1, sample_ann_dict, dclass)
                    create_group_data(group=group, pc=pc0, gm=is_ground_0.astype(np.bool_), pose=pose0.astype(np.float32),
                                    flow_0to1=scene_flow['flow_0_1'], flow_valid=scene_flow['valid_0'], flow_category=scene_flow['classes_0'], \
                                    flow_instance=scene_flow['flow_instance_id'],
                                    ego_motion=scene_flow['ego_motion'].astype(np.float32))
                                    

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
    # for x in tqdm(args):
    #     proc(x, ignore_current_process=True)
    #     break

    if nproc <= 1:
        for x in tqdm(args):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(scene_list), ncols=100))

def main(
    data_dir: str = "/home/kin/data/nus/raw",
    mode: str = "v1.0-mini",
    output_dir: str ="/home/kin/data/nus/h5py/demo",
    nproc: int = (multiprocessing.cpu_count() - 1),
    only_index: bool = False,
    split_name = None
):
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini'] # defined by nus.
    assert mode in available_vers
    # nusc = NuScenes(dataroot=data_dir, version=mode, verbose=False)
    # print(f"Processing {mode} dataset with {len(nusc.scene)} scenes...")
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