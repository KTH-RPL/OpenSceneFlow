"""
# Created: 2024-07-07 22:18
# Copyright (C) 2024-now, Scania Sverige EEARP Group
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# License: GPLv2, allow it free only for academic use.
# Description: Preprocess Data, save as h5df format for faster loading
"""

import fire, time, os, sys, json, h5py
from pathlib import Path
from multiprocessing import current_process
from tqdm import tqdm
import multiprocessing
import numpy as np
from typing import Optional

# import zod
from zod.constants import Camera, Lidar
from zod.data_classes.sequence import ZodSequence
from zod._zod_dataset import _create_frame

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..'))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index

from linefit import ground_seg
GROUNDSEG_config = f"{BASE_DIR}/conf/others/zod.toml"

def process_log(data_dir: Path, log_id: str, output_dir: Path, n: Optional[int] = None) :

    def create_group_data(group, pc, pc_dt, pose, pc_id, gm=None, flow_0to1=None, flow_valid=None, flow_category=None, ego_motion=None):
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('lidar_id', data=pc_id.astype(np.uint8)) # sensor id
        group.create_dataset('lidar_dt', data=pc_dt.astype(np.float32)) # deltaT
        group.create_dataset('pose', data=pose.astype(np.float32))
        if gm is not None:
            group.create_dataset('ground_mask', data=gm.astype(np.bool_))
        if flow_0to1 is not None:
            # ground truth flow information
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))

    with open(data_dir / log_id / 'info.json', "r") as f:
        frames = _create_frame(json.load(f), data_dir.as_posix().replace('drives', ''))
    seq = ZodSequence(frames)
    mygroundseg = ground_seg(GROUNDSEG_config)
    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, frame in enumerate(seq.info.get_camera_lidar_map()):
            
            # NOTE(Qingwen): for a quick check downsampled data.. otherwise one h5py might have really long frames.
            # FIXME(Qingwen): maybe split too long scene to mini-scene etc later?
            if cnt % 4 != 0:
                continue

            camera_frame, lidar_frame = frame
            # img = camera_frame.read()
            pcd = seq.get_compensated_lidar(camera_frame.time)
            pose = seq.ego_motion.get_poses(camera_frame.time.timestamp())

            diode_idx = pcd.diode_idx
            lidar_id = np.zeros(diode_idx.shape, dtype=np.uint8)

            lidar_id[(diode_idx >= 0) & (diode_idx < 128)] = 1
            lidar_id[(diode_idx >= 128) & (diode_idx < 144)] = 2
            lidar_id[(diode_idx >= 144) & (diode_idx < 160)] = 3
            ego2sensor = seq._calibration.lidars[Lidar.VELODYNE].extrinsics
            pose = pose @ ego2sensor.transform
            lidar_timestamp = pcd.core_timestamp
            lidar_dt = pcd.timestamps - lidar_timestamp
            group = f.create_group(str(int(camera_frame.time.timestamp() * 10e6)))
            
            # NOTE(Qingwen): only need 128-channel long-range lidar, while if you want feel free to comment these.
            # points = pcd.points # for all three lidars (1 long, 2 really short)
            points = pcd.points[lidar_id == 1]
            lidar_dt = lidar_dt[lidar_id == 1]
            lidar_id = lidar_id[lidar_id == 1]
            ground_mask = mygroundseg.run(points[:, :3])

            create_group_data(group, points, lidar_dt, pose.astype(np.float32), lidar_id, gm=np.array(ground_mask, dtype=np.bool_))

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
    # like here only 000018 scene.
    logs = ['000018']
    args = sorted([(data_dir, log, output_dir) for log in logs])
    print(f'Using {nproc} processes to process data: {data_dir} to .h5 format. (#scenes: {len(args)})')
    # for debug
    for x in tqdm(args):
        print(x)
        proc(x, ignore_current_process=True)
        break
    
    # comment out if you want to process all scene.
    # if nproc <= 1:
    #     for x in tqdm(args, ncols=120):
    #         proc(x, ignore_current_process=True)
    # else:
    #     with Pool(processes=nproc) as p:
    #         res = list(tqdm(p.imap_unordered(proc, args), total=len(logs), ncols=120))

def main(
    dataset_root: str = "/home/kin/DATA_HDD/public_data/zod/drives",
    output_dir: str ="/home/kin/data/zod/h5py/himo",
    nproc: int = (multiprocessing.cpu_count() - 1),
    only_index: bool = False,
):
    output_dir_ = Path(output_dir)
    if only_index:
        create_reading_index(output_dir_)
        return
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(Path(dataset_root), output_dir_, nproc)
    create_reading_index(output_dir_)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nRunning {__file__} used: {(time.time() - start_time)/60:.2f} mins")