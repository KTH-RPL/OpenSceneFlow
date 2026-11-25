"""
# Created: 2026-11-22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), AI Assistant (Claude Code)
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.
#
# Description: Waymo Open Dataset v2.0.1 (parquet) -> h5 with scene flow from the tracked lidar boxes,
#   the same way as extract_av2.py: points inside box k at t0 move with box k from t0 to t1,
#   the other points move with the ego motion. Points in a box without a match at t1 are not valid.
#   Same conventions as extract_waymo.py (v1.4 + flow labels): TOP lidar first return only,
#   z moved to the TOP lidar height, category ids mapped to the av2 ones.
#
# env need (py3.10): pip install waymo-open-dataset-tf-2-12-0==1.6.4 h5py fire tqdm linefit
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import multiprocessing
from multiprocessing import Pool, current_process
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import fire, time, h5py
from tqdm import tqdm

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils

import sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index, check_h5py_file_exists
from linefit import ground_seg
GROUNDSEG_config = f"{BASE_DIR}/conf/others/waymo.toml"

TOP = 1 # LaserName.TOP
BOUNDING_BOX_EXPANSION = 0.2
# waymo box type -> av2 category index (same as extract_waymo.py)
# 0: unknown, 1: vehicle -> 19 REGULAR_VEHICLE, 2: pedestrian -> 17 PEDESTRIAN, 3: sign -> 21 SIGN, 4: cyclist -> 4 BICYCLIST
WAYMO2AV2 = {0: 0, 1: 19, 2: 17, 3: 21, 4: 4}

def read(data_dir: Path, component: str, scene: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / component / f'{scene}.parquet')

def box_SE3(row):
    """4x4 object->vehicle transform of a lidar box row (heading = yaw around z)."""
    c, s = np.cos(row['[LiDARBoxComponent].box.heading']), np.sin(row['[LiDARBoxComponent].box.heading'])
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [row['[LiDARBoxComponent].box.center.x'], row['[LiDARBoxComponent].box.center.y'], row['[LiDARBoxComponent].box.center.z']]
    return T

def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]

def interior(T_obj, size, pts):
    """Mask of the points (vehicle frame) inside a box with object->vehicle transform T_obj and size (l, w, h)."""
    local = transform(np.linalg.inv(T_obj), pts)
    return np.all(np.abs(local) <= np.asarray(size) / 2, axis=1)

def compute_flow(pc0, lidar0, boxes0, boxes1, ego1_SE3_ego0, dclass):
    """pc0: points in the vehicle frame at t0, lidar0: the same points as saved (z moved to the sensor height),
    boxes{0,1}: {object_id: row}. Returns the flow of lidar0 to t1.
    NOTE: the masks are computed in the vehicle frame (boxes are there), the transforms are applied to the saved points,
    so the flow of static points is exactly ego_motion applied to the saved lidar, same as extract_waymo.py."""
    flow = transform(ego1_SE3_ego0, lidar0) - lidar0
    valid = np.ones(len(pc0), dtype=np.bool_)
    classes = np.zeros(len(pc0), dtype=np.uint8)
    instances = np.zeros(len(pc0), dtype=np.int16)
    for oid, b0 in boxes0.items():
        T0 = box_SE3(b0)
        size = np.array([b0['[LiDARBoxComponent].box.size.x'], b0['[LiDARBoxComponent].box.size.y'], b0['[LiDARBoxComponent].box.size.z']])
        mask = interior(T0, size, pc0)
        if oid not in boxes1:
            valid[mask] = 0
            continue
        T1 = box_SE3(boxes1[oid])
        c1_SE3_c0 = T1 @ np.linalg.inv(T0) # vehicle t0 -> vehicle t1, moving with the object
        # NOTE(HiMo): expand the box of a moving object, same as extract_av2.py
        rel = transform(np.linalg.inv(ego1_SE3_ego0) @ c1_SE3_c0, pc0[mask]) - pc0[mask]
        delta_move = abs(np.linalg.norm(rel, axis=0).mean()) if mask.any() else 0.0
        if delta_move > 0.04:
            size = size + np.array([BOUNDING_BOX_EXPANSION + min(delta_move/2, 2), BOUNDING_BOX_EXPANSION, BOUNDING_BOX_EXPANSION])
            mask = interior(T0, size, pc0)
        flow[mask] = transform(c1_SE3_c0, lidar0[mask]) - lidar0[mask]
        classes[mask] = WAYMO2AV2.get(int(b0['[LiDARBoxComponent].type']), 0)
        instances[mask] = dclass[oid] + 1
    return flow.astype(np.float32), valid, classes, instances

def process_log(data_dir: Path, scene: str, output_dir: Path, n=None):
    lidar_df = read(data_dir, 'lidar', scene)
    lidar_df = lidar_df[lidar_df['key.laser_name'] == TOP]
    lidar_pose_df = read(data_dir, 'lidar_pose', scene)
    lidar_pose_df = lidar_pose_df[lidar_pose_df['key.laser_name'] == TOP]
    vpose_df = read(data_dir, 'vehicle_pose', scene)
    calib_df = read(data_dir, 'lidar_calibration', scene)
    calib = v2.LiDARCalibrationComponent.from_dict(calib_df[calib_df['key.laser_name'] == TOP].iloc[0])
    sensor_z = np.reshape(np.array(calib.extrinsic.transform), [4, 4])[2, 3]
    box_df = read(data_dir, 'lidar_box', scene)

    df = v2.merge(v2.merge(lidar_df, lidar_pose_df), vpose_df).sort_values('key.frame_timestamp_micros')
    timestamps = df['key.frame_timestamp_micros'].tolist()
    if check_h5py_file_exists(output_dir/f'{scene}.h5', timestamps):
        return

    boxes = defaultdict(dict) # ts -> {object_id: row}
    for _, row in box_df.iterrows():
        boxes[row['key.frame_timestamp_micros']][row['key.laser_object_id']] = row

    frames = []
    for _, row in df.iterrows():
        lidar = v2.LiDARComponent.from_dict(row)
        pixel_pose = v2.LiDARPoseComponent.from_dict(row)
        vpose = v2.VehiclePoseComponent.from_dict(row)
        pc = lidar_utils.convert_range_image_to_point_cloud(
            lidar.range_image_return1, calib, pixel_pose.range_image_return1, vpose, keep_polar_features=False).numpy()
        pose = np.reshape(np.array(vpose.world_from_vehicle.transform), [4, 4])
        frames.append((row['key.frame_timestamp_micros'], pc.astype(np.float64), pose))

    dclass = defaultdict(lambda: len(dclass))
    mygroundseg = ground_seg(GROUNDSEG_config)
    with h5py.File(output_dir/f'{scene}.h5', 'a') as f:
        for i, (ts0, pc0, pose0) in enumerate(frames):
            if pc0.shape[0] < 256:
                print(f'{scene}/{ts0} has less than 256 points, skip.')
                continue
            lidar = pc0.copy()
            lidar[:, 2] -= sensor_z # move center to the sensor height, same as extract_waymo.py
            gm = mygroundseg.run(lidar[:, :3]) | (lidar[:, 2] < -1.8)
            group = f.create_group(str(ts0))
            group.create_dataset('lidar', data=lidar.astype(np.float32))
            group.create_dataset('pose', data=pose0.astype(np.float64))
            group.create_dataset('ground_mask', data=np.array(gm).astype(bool))
            if i == len(frames) - 1:
                continue
            ts1, _, pose1 = frames[i+1]
            ego1_SE3_ego0 = np.linalg.inv(pose1) @ pose0
            group.create_dataset('ego_motion', data=ego1_SE3_ego0.astype(np.float32))
            if ts0 not in boxes or ts1 not in boxes:
                continue
            flow, valid, classes, instances = compute_flow(pc0, lidar, boxes[ts0], boxes[ts1], ego1_SE3_ego0, dclass)
            group.create_dataset('flow', data=flow)
            group.create_dataset('flow_is_valid', data=valid)
            group.create_dataset('flow_category_indices', data=classes)
            group.create_dataset('flow_instance_id', data=instances)

def proc(x, ignore_current_process=False):
    pos = 1 if ignore_current_process else current_process()._identity[0]
    try:
        process_log(*x, n=pos)
    except Exception as e:
        print(f'\n--- ERROR [{x[1]}]: {e}')

def main(
    data_dir: str = "/home/kin/data/waymo/2.0.1", # with training/ validation/ inside, each with the component folders (lidar, lidar_box, ...)
    mode: str = "validation",
    output_dir: str = "/home/kin/data/waymo/h5py",
    nproc: int = (multiprocessing.cpu_count() - 1),
    scene_range: list = [-1, -1], # [start, end) index in the sorted scene list, for slurm arrays / debug
    only_index: bool = False,
):
    data_dir_ = Path(data_dir) / mode
    output_dir_ = Path(output_dir) / {'training': 'train', 'validation': 'valid'}.get(mode, mode)
    if not only_index:
        output_dir_.mkdir(exist_ok=True, parents=True)
        logs = sorted(p.stem for p in (data_dir_ / 'lidar').glob('*.parquet'))
        if scene_range[0] != -1:
            logs = logs[scene_range[0]:scene_range[1]]
        args = [(data_dir_, log, output_dir_) for log in logs]
        print(f'Using {nproc} processes to process {data_dir_} to {output_dir_} (#scenes: {len(args)})')
        if nproc <= 1:
            for x in tqdm(args, ncols=120):
                proc(x, ignore_current_process=True)
        else:
            with Pool(processes=nproc) as p:
                list(tqdm(p.imap_unordered(proc, args), total=len(args), ncols=120))
    if scene_range[0] == -1 or only_index:
        create_reading_index(output_dir_)
        create_reading_index(output_dir_, flow_inside_check=True)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")
