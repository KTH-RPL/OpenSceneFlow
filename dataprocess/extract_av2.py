"""
# Created: 2023-11-01 17:02
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Preprocess Data, save as h5df format for faster loading
# Reference: 
#   * ZeroFlow data preprocessing work: https://github.com/kylevedder/argoverse2-sf
#   * Argoverse API source code: https://github.com/argoverse/av2-api
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3
from av2.structures.sweep import Sweep
from av2.structures.cuboid import CuboidList, Cuboid
from av2.utils.io import read_feather
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3
from av2.datasets.sensor.constants import AnnotationCategories

import multiprocessing
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional, Tuple, Dict, Union, Final
from tqdm import tqdm
import numpy as np
import fire, time, h5py
from collections import defaultdict
import pickle
from zipfile import ZipFile
import pandas as pd
from copy import deepcopy

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index
from src.utils.av2_eval import read_ego_SE3_sensor

BOUNDING_BOX_EXPANSION: Final = 0.2
CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}

def create_eval_mask(data_mode: str, output_dir_: Path, mask_dir: str):
    """
    Need download the official mask file run: `s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .`
    Check more in our assets/README.md
    """
    mask_file_path = Path(mask_dir) / f"{data_mode}-masks.zip"
    if not mask_file_path.exists():
        print(f'{mask_file_path} not found, please download the mask file for official evaluation.')
        return
    # extract the mask file
    with ZipFile(mask_file_path, 'r') as zipObj:
        zipObj.extractall(Path(mask_dir) / f"{data_mode}-masks")
    
    data_index = []
    # list scene ids
    scene_ids = os.listdir(Path(mask_dir) / f"{data_mode}-masks")
    for scene_id in tqdm(scene_ids, desc=f'Create {data_mode} eval mask', ncols=100):
        timestamps = sorted([int(file.replace('.feather', ''))
                        for file in os.listdir(Path(mask_dir) / f"{data_mode}-masks" / scene_id)
                        if file.endswith('.feather')])
        if not os.path.exists(output_dir_ / f'{scene_id}.h5'):
            continue
        with h5py.File(output_dir_ / f'{scene_id}.h5', 'r+') as f:
            for ts in timestamps:
                key = str(ts)
                if key not in f.keys():
                    print(f'{scene_id}/{key} not found')
                    continue
                group = f[key]
                mask = pd.read_feather(Path(mask_dir) / f"{data_mode}-masks" / scene_id / f"{key}.feather").to_numpy().astype(bool)
                group.create_dataset('eval_mask', data=mask)
                data_index.append([scene_id, key])

    with open(output_dir_/'index_eval.pkl', 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create reading index Successfully")

def read_pose_pc_ground(data_dir: Path, log_id: str, timestamp: int, avm: ArgoverseStaticMap):
    log_poses_df = read_feather(data_dir / log_id / "city_SE3_egovehicle.feather")
    # more detail: https://argoverse.github.io/user-guide/datasets/lidar.html#sensor-suite
    ego2sensor_pose = read_ego_SE3_sensor((data_dir / log_id))['up_lidar']
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin([timestamp])]
    pose = convert_pose_dataframe_to_SE3(filtered_log_poses_df.loc[filtered_log_poses_df["timestamp_ns"] == timestamp])
    pc = Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{timestamp}.feather").xyz
    # transform to city coordinate since sweeps[0].xyz is in ego coordinate to get ground mask
    is_ground = avm.get_ground_points_boolean(pose.transform_point_cloud(pc))

    # NOTE(SeFlow): transform to sensor coordinate, since some ray-casting based methods need sensor coordinate
    pc = ego2sensor_pose.inverse().transform_point_cloud(pc) 
    return pc, pose, is_ground

def compute_sceneflow(data_dir: Path, log_id: str, timestamps: Tuple[int, int]) -> Dict[str, Union[np.ndarray, SE3]]:
    """Compute sceneflow between the sweeps at the given timestamps.
        Args:
          data_dir: Argoverse 2.0 directory, e.g. /home/kin/data/av2/sensor/train
          log_id: unique id.
          timestamps: the timestamps of the lidar sweeps to compute flow between
        Returns:
          Dictionary with fields:
            pcl_0: Nx3 array containing the points at time 0
            pcl_1: Mx3 array containing the points at time 1
            flow_0_1: Nx3 array containing flow from timestamp 0 to 1
            flow_1_0: Mx3 array containing flow from timestamp 1 to 0
            valid_0: Nx1 array indicating if the returned flow from 0 to 1 is valid (1 for valid, 0 otherwise)
            valid_1: Mx1 array indicating if the returned flow from 1 to 0 is valid (1 for valid, 0 otherwise)
            classes_0: Nx1 array containing the class ids for each point in sweep 0
            classes_1: Nx1 array containing the class ids for each point in sweep 0
            pose_0: SE3 pose at time 0
            pose_1: SE3 pose at time 1
            ego_motion: SE3 motion from sweep 0 to sweep 1
    """
    def compute_flow(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        
        flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz) -  sweeps[0].xyz
        # Convert to float32s
        flow = flow.astype(np.float32)
        
        valid = np.ones(len(sweeps[0].xyz), dtype=np.bool_)
        # classes = -np.ones(len(sweeps[0].xyz), dtype=np.int8)
        classes = np.zeros(len(sweeps[0].xyz), dtype=np.uint8)

        # # old version
        # for id in cuboids[0]:
        #     c0 = cuboids[0][id]
        #     c0.length_m += BOUNDING_BOX_EXPANSION # the bounding boxes are a little too tight and some points are missed
        #     c0.width_m += BOUNDING_BOX_EXPANSION
        #     obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
        #     classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
        
        #     if id in cuboids[1]:
        #         c1 = cuboids[1][id]
        #         c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
        #         obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
        #         flow[obj_mask] = obj_flow.astype(np.float32)
        #     else:
        #         valid[obj_mask] = 0

        # NOTE(HiMo): box expansion based on the object velocity
        # check more detail: https://kin-zhang.github.io/HiMo
        for id in cuboids[0]:
            c0 = deepcopy(cuboids[0][id])
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse()))
                rel_obj_flow = c1_SE3_c0_ego_frame.transform_point_cloud(obj_pts) - obj_pts
                delta_move = abs(np.linalg.norm(rel_obj_flow, axis=0).mean())

                if delta_move > 0.04: # only when it's moving
                    c0 = cuboids[0][id]
                    c0.length_m += (BOUNDING_BOX_EXPANSION + min(delta_move/2, 2)) # since 180/360 for two LiDARs orientation
                    c0.width_m += BOUNDING_BOX_EXPANSION
                    c0.height_m += BOUNDING_BOX_EXPANSION
                obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)

                # NOTE(Qingwen): after expansion, we need to recompute the flow
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
                flow[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid[obj_mask] = 0
        return flow, classes, valid, ego1_SE3_ego0
    sweeps = [Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{ts}.feather") for ts in timestamps]

    # ================== Load annotations ==================
    annotations_feather_path = data_dir / log_id / "annotations.feather"
    
    if not annotations_feather_path.exists():
        # print(f'{annotations_feather_path} not found')
        timestamp_cuboid_index = {}
    else:
        # Load annotations from disk.
        # NOTE: This file contains annotations for the ENTIRE sequence.
        # The sweep annotations are selected below.
        cuboid_list = CuboidList.from_feather(annotations_feather_path)

        raw_data = read_feather(annotations_feather_path)
        ids = raw_data.track_uuid.to_numpy()
        timestamp_cuboid_index = defaultdict(dict)
        for id, cuboid in zip(ids, cuboid_list.cuboids):
            timestamp_cuboid_index[cuboid.timestamp_ns][id] = cuboid
    # ================== Load annotations ==================

    cuboids = [timestamp_cuboid_index.get(ts, {}) for ts in timestamps]

    log_poses_df = read_feather(data_dir / log_id / "city_SE3_egovehicle.feather")
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin(timestamps)]
    poses = [convert_pose_dataframe_to_SE3(filtered_log_poses_df.loc[filtered_log_poses_df["timestamp_ns"] == ts]) for ts in timestamps]

    flow_0_1, classes_0, valid_0, ego_motion = compute_flow(sweeps, cuboids, poses)

    return {'pcl_0': sweeps[0].xyz, 'pcl_1' :sweeps[1].xyz, 'flow_0_1': flow_0_1,
            'valid_0': valid_0, 'classes_0': classes_0, 
            'pose_0': poses[0], 'pose_1': poses[1],
            'ego_motion': ego_motion}

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

    log_map_dirpath = data_dir / log_id / "map"
    if(len(os.listdir(log_map_dirpath))<3):
        print(f'{log_map_dirpath} needed by 3 to find the ground layer, check if you are using the correct *sensor* dataset')
        print("If you are using *lidar* dataset, Please run the following command to generate the map files:")
        print(f"python run_steps/0_additional_lidar_map.py --argo_dir {data_dir}")
        return
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    timestamps = sorted([int(file.replace('.feather', ''))
                        for file in os.listdir(data_dir / log_id / "sensors/lidar")
                        if file.endswith('.feather')])

    gt_flow_flag = False if not (data_dir / log_id / "annotations.feather").exists() else True

    # if n is not None:
    #     iter_bar = tqdm(zip(timestamps, timestamps[1:]), leave=False,
    #                      total=len(timestamps) - 1, position=n,
    #                      desc=f'Log {log_id}')
    # else:
    #     iter_bar = zip(timestamps, timestamps[1:])

    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, ts0 in enumerate(timestamps):
            group = f.create_group(str(ts0))
            pc0, pose0, is_ground_0 = read_pose_pc_ground(data_dir, log_id, ts0, avm)
            if pc0.shape[0] < 256:
                print(f'{log_id}/{ts0} has less than 256 points, skip this scenarios. Please check the data if needed.')
                break
            if cnt == len(timestamps) - 1 or not gt_flow_flag:
                create_group_data(group, pc0, is_ground_0.astype(np.bool_), pose0.transform_matrix.astype(np.float32))
            else:
                ts1 = timestamps[cnt + 1]
                scene_flow = compute_sceneflow(data_dir, log_id, (ts0, ts1))
                create_group_data(group, pc0, is_ground_0.astype(np.bool_), pose0.transform_matrix.astype(np.float32),
                                  scene_flow['flow_0_1'], scene_flow['valid_0'], scene_flow['classes_0'],
                                  scene_flow['ego_motion'].transform_matrix.astype(np.float32))

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(data_dir: Path, output_dir: Path, nproc: int):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
       Args:
         data_dir: Argoverse 2.0 directory
         output_dir: Output directory.
    """
    
    if not data_dir.exists():
        print(f'{data_dir} not found')
        return
    
    # NOTE(Qingwen): if you don't want to all data_dir, then change here: logs = logs[:10] only 10 scene.
    logs = os.listdir(data_dir)
    args = sorted([(data_dir, log, output_dir) for log in logs])
    print(f'Using {nproc} processes to process data: {data_dir} to .h5 format. (#scenes: {len(args)})')
    # for debug
    # for x in tqdm(args):
    #     proc(x, ignore_current_process=True)
    #     break
    if nproc <= 1:
        for x in tqdm(args, ncols=120):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(logs), ncols=120))

def main(
    argo_dir: str = "/home/kin/data/av2",
    output_dir: str ="/home/kin/data/av2/preprocess",
    av2_type: str = "sensor",
    data_mode: str = "val",
    mask_dir: str = "/home/kin/data/av2/3d_scene_flow",
    nproc: int = (multiprocessing.cpu_count() - 1),
    only_index: bool = False,
):
    data_root_ = Path(argo_dir) / av2_type/ data_mode
    output_dir_ = Path(output_dir) / av2_type / data_mode
    if only_index:
        create_reading_index(output_dir_)
        return
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(data_root_, output_dir_, nproc)
    create_reading_index(output_dir_)
    if data_mode == "val" or data_mode == "test":
        create_eval_mask(data_mode, output_dir_, mask_dir)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")