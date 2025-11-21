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
* python tools/visualization_rerun.py --data_dir /home/kin/data/av2/h5py/demo/train --res_name "['flow','deflow']"
# 
"""

import numpy as np
import fire, time
from tqdm import tqdm
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, flow_to_rgb
from src.utils.o3d_view import color_map
import rerun as rr
import rerun.blueprint as rrb
import argparse

def main(
    data_dir: str ="/home/kin/data/av2/h5py/demo/train",
    res_name: list = ["flow"],
    vis_interval: int = 1,
    start_id: int = 0,
    point_size: float = 0.25,
    tone: str = 'dark',
):
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)
    if len(dataset) > 500 and vis_interval < 5:
        print(f"Total {len(dataset)} data in {data_dir}, we suggest to only visualize a subset of them.")
        print(f"or set `vis_interval` to a larger value, e.g., 10, 20, 50, 100, ...")
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

    for data_id in (pbar := tqdm(range(start_id, len(dataset)))):
        if data_id % vis_interval != 0:
            continue

        rr.set_time_sequence('frame_idx', data_id)

        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        
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
                if mode in data:
                    labels = data[mode]
                    for label_i in np.unique(labels):
                        if label_i > 0:
                            flow_color[labels == label_i] = color_map[label_i % len(color_map)]

                # log flow mode 
                rr.log(f"world/ego_vehicle/lidar/{mode}", rr.Points3D(pc0[:,:3], colors=flow_color, radii=np.ones((pc0.shape[0],))*point_size/2))

            elif mode in data:
                flow = data[mode] - pose_flow # ego motion compensation here.
                flow_nanmask = np.isnan(data[mode]).any(axis=1)
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
