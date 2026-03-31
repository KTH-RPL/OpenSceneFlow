"""
# Created: 2024-11-20 22:30
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Ajinkya Khoche  (https://ajinkyakhoche.github.io/)
#         Qingwen Zhang (https://kin-zhang.github.io/)
# 
# Description: view scene flow dataset after preprocess or evaluation. Dependence on `mamba install rerun-sdk` or `pip install rerun-sdk`.
# 
# 
# Usage with demo data: (flow is ground truth flow, `other_name` is the estimated flow from the model)
* python tools/visualization_rerun.py --scene_file /home/kin/data/av2/h5py/demo/val/25e5c600-36fe-3245-9cc0-40ef91620c22.h5 --res_name "['flow','deflow']"
# 
"""

import numpy as np
import fire, time
from tqdm import tqdm
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import flow_to_rgb
from src.utils.o3d_view import color_map
import rerun as rr
import rerun.blueprint as rrb
import argparse
import h5py

def main(
    scene_file: str ="/home/kin/data/av2/h5py/demo/val/25e5c600-36fe-3245-9cc0-40ef91620c22.h5",
    res_name: list = ["flow"],
    point_size: float = 0.25,
    max_distance: float = 35.0,
    tone: str = 'dark',
):
    if not os.path.exists(scene_file):
        print(f"File {scene_file} not found.")
        return
        
    f = h5py.File(scene_file, 'r')
    keys = sorted(list(f.keys()), key=lambda x: int(x))
    data_len = len(keys) - 1

    if data_len <= 0:
        print(f"Not enough frames in {scene_file} for flow visualization.")
        return

    background_color = (255, 255, 255) if tone == 'bright' else (80, 90, 110)

    # setup the rerun environment
    blueprint = rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D",
                origin="world",
                # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
                defaults=[rr.components.ImagePlaneDistance(4.0)],
                background=background_color,
                overrides={"world/ego_vehicle": [rr.components.AxisLength(4.0)]},
            ),
            column_shares=[3, 1],
        ),
    )
    # fake args
    rr.script_setup(
        args=argparse.Namespace(
            # headless=False,
            # connect=False,
            serve=True,
            # addr=None,
            # save=None,
            stdout=False,
        ), application_id="OpenSceneFlow Visualization",default_blueprint=blueprint)
    
    if tone == 'light':
        pcd_color = [0.25, 0.25, 0.25]
        ground_color = [0.75, 0.75, 0.75]
    elif tone == 'dark':
        pcd_color = [1., 1., 1.]
        ground_color = [0.25, 0.25, 0.25]

    now_scene_id = os.path.basename(scene_file).replace('.h5', '')
    for data_id in (pbar := tqdm(range(0, data_len))):
        rr.set_time_sequence('frame_idx', data_id)

        timestamp = keys[data_id]
        next_timestamp = keys[data_id + 1]
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {timestamp}")

        group = f[timestamp]
        next_group = f[next_timestamp]

        pc0 = group['lidar'][:][:,:3]
        gm0 = group['ground_mask'][:]
        pose0 = group['pose'][:]
        pose1 = next_group['pose'][:]
        
        dist_mask = (np.abs(pc0[:, 0]) < max_distance) & (np.abs(pc0[:, 1]) < max_distance) & (np.abs(pc0[:, 2]) < max_distance)
        pc0 = pc0[dist_mask]
        gm0 = gm0[dist_mask]
        
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]           
        
        # log ego pose
        rr.log(
            f"world/ego_vehicle",
            rr.Transform3D(
                translation=np.zeros((3,)),
                rotation=rr.Quaternion(xyzw=np.array([0,0,0,1])),
                from_parent=False,
            ),
            static=True,
        )

        for mode in res_name:
            flow_color = np.tile(pcd_color, (pc0.shape[0], 1))
            flow_color[gm0] = ground_color

            if mode in ['dufo', 'label']:
                if mode in group:
                    labels = group[mode][:]
                    labels = labels[dist_mask]
                    for label_i in np.unique(labels):
                        if label_i > 0:
                            flow_color[labels == label_i] = color_map[label_i % len(color_map)]

                # log flow mode 
                rr.log(f"world/ego_vehicle/lidar/{mode}", rr.Points3D(pc0[:,:3], colors=flow_color, radii=np.ones((pc0.shape[0],))*point_size/2))

            elif mode in group:
                flow_data = group[mode][:]
                flow_data = flow_data[dist_mask]
                flow = flow_data - pose_flow # ego motion compensation here.
                flow_nanmask = np.isnan(flow_data).any(axis=1)
                flow_color = np.tile(pcd_color, (pc0.shape[0], 1))
                flow_color[~flow_nanmask,:] = flow_to_rgb(flow[~flow_nanmask,:]) / 255.0
                flow_color[gm0] = ground_color

                # log flow mode with labels
                labels = ["flow={:.2f},{:.2f},{:.2f}".format(fx,fy,fz) for fx,fy,fz in flow.round(2)]
                rr.log(f"world/ego_vehicle/lidar/{mode}", rr.Points3D(pc0[:,:3], colors=flow_color, radii=np.ones((pc0.shape[0],))*point_size/2, labels=labels))

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")
