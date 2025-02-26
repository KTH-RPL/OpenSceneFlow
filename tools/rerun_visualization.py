"""
# Created: 2024-11-20 22:30
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Ajinkya Khoche  (https://ajinkyakhoche.github.io/)
#
# 
# Description: view scene flow dataset after preprocess or evaluation. 
# --flow_mode accepts a list eg. ["flow", "flow_est", ...] where "flow" is ground truth and "flow_est" is estimated from a neural network (result of save.py script).

# Usage with demo data: (flow is ground truth flow, `other_name` is the estimated flow from the model)
* python tools/rerun_visualization.py --data_dir /home/kin/data/av2/preprocess_v2/demo/sensor/val --flow_mode ['flow', 'deflow' , 'ssf'] 
"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, flow_to_rgb
from src.utils.o3d_view import color_map
import rerun as rr
import rerun.blueprint as rrb
import argparse

VIEW_FILE = f"{BASE_DIR}/assets/view/av2.json"
DESCRIPTION = """
Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
Author: [Ajinkya Khoche](https://ajinkyakhoche.github.io/)
Visualize scene flow dataset including lidar.

The code is modified from rerun example
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/nuscenes_dataset).
"""

def vis_rerun(data_dir, flow_mode, start_id, vis_interval, point_size, tone):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True)

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

        for mode in flow_mode:
            flow_color = np.tile(pcd_color, (pc0.shape[0], 1))
            flow_color[gm0] = ground_color

            if mode in ['dufo_label', 'label']:
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

        # log dynamic mask
        if 'pc0_dynamic_mask' in data:
            pc0_dynamic_mask = data['pc0_dynamic_mask']
            flow_color_dynamic = np.tile(pcd_color, (pc0.shape[0], 1))
            flow_color_dynamic[gm0] = ground_color
            flow_color_dynamic[pc0_dynamic_mask] = color_map[1]
            rr.log(f"world/ego_vehicle/lidar/dynamic", rr.Points3D(pc0[~gm0,:3], colors=flow_color_dynamic[~gm0,:], radii=np.ones((pc0[~gm0,:].shape[0],))*point_size/4))


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Visualizes flow dataset using the Rerun SDK.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ajinkya/data/av2/preprocess/sensor/vis",
        help="data directory to preprocess",
    )
    parser.add_argument(
        "--flow_mode",
        type=list,
        default=["flow"], # ["flow", "flow_est", "label", "dufo_label"]
        help="flow modes to visualize",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="start id to visualize",
    )
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=1,
        help="Optional: visualize every x steps",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.25,
        help="point size",
    )
    parser.add_argument(
        "--tone",
        type=str,
        default="light",
        help="tone of the visualization",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    if args.tone == 'light':
        background_color = (255, 255, 255)
    else:
        background_color = (80, 90, 110)

    # setup the rerun environment
    blueprint = rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D",
                origin="world",
                # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
                defaults=[rr.components.ImagePlaneDistance(4.0)],
                background=background_color,
                # Transform arrows for the vehicle shouldn't be too long.
                overrides={"world/ego_vehicle": [rr.components.AxisLength(5.0)]},
            ),
            rrb.TextDocumentView(origin="description", name="Description"),
            column_shares=[3, 1],
        ),
    )

    rr.script_setup(args, "rerun_vis", default_blueprint=blueprint)

    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        timeless=True,
    )

    # call the main function
    vis_rerun(args.data_dir, args.flow_mode, args.start_id, args.vis_interval, args.point_size, args.tone)
    print(f"Time used: {time.time() - start_time:.2f} s")
