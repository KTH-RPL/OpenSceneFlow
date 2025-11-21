"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Ajinkya Khoche (https://ajinkyakhoche.github.io/)
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: view scene flow dataset after preprocess.

# CHANGELOG:
# 2024-09-10 (Ajinkya): Add vis_multiple(), to visualize multiple flow modes at once.

# Usage: (flow is ground truth flow, `other_name` is the estimated flow from the model)
* python tools/visualization.py --data_dir /home/kin/data/av2/h5py/demo/train --res_name 'flow' --mode vis
* python tools/visualization.py --data_dir /home/kin/data/av2/h5py/demo/train --res_name "['flow', 'deflow' , 'ssf']" --mode mul

"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, flow_to_rgb
from src.utils.o3d_view import MyVisualizer, MyMultiVisualizer, color_map, create_bev_square


VIEW_FILE = f"{BASE_DIR}/assets/view/demo.json"

def check_flow(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow", # "flow", "flow_est"
    start_id: int = 0,
    point_size: float = 3.0,
):
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if res_name == 'flow' else f'{res_name} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(start_id, len(dataset)))):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        gm0 = data['gm0']
        pc0 = data['pc0'][~gm0]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.paint_uniform_color([1.0, 0.0, 0.0]) # red: pc0

        pc1 = data['pc1']
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3][~data['gm1']])
        pcd1.paint_uniform_color([0.0, 1.0, 0.0]) # green: pc1

        pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + pose_flow) # if you want to check pose_flow
        pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + data[res_name][~gm0])
        pcd2.paint_uniform_color([0.0, 0.0, 1.0]) # blue: pc0 + flow
        o3d_vis.update([pcd, pcd1, pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

def vis(
    data_dir: str ="/home/kin/data/av2/h5py/demo/val",
    res_name: str = "flow", # any res_name we write before in HDF5Data
    start_id: int = 0,
    point_size: float = 2.0,
    mode: str = "vis",
):
    if mode != "vis":
        return
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if res_name == 'flow' else f'{res_name} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    # opt.background_color = np.asarray([216, 216, 216]) / 255.0
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    # opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(start_id, len(dataset)))):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0

        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        if res_name == 'raw': # no result, only show **raw point cloud**
            pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
            pcd.paint_uniform_color([1.0, 1.0, 1.0])
        elif res_name in ['dufo', 'label']:
            labels = data[res_name]
            pcd_i = o3d.geometry.PointCloud()
            for label_i in np.unique(labels):
                pcd_i.points = o3d.utility.Vector3dVector(pc0[labels == label_i][:, :3])
                if label_i <= 0:
                    pcd_i.paint_uniform_color([1.0, 1.0, 1.0])
                else:
                    pcd_i.paint_uniform_color(color_map[label_i % len(color_map)])
                pcd += pcd_i
        elif res_name in data:
            pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
            flow = data[res_name] - pose_flow # ego motion compensation here.
            flow_color = flow_to_rgb(flow) / 255.0
            is_dynamic = np.linalg.norm(flow, axis=1) > 0.1
            flow_color[~is_dynamic] = [1, 1, 1]
            flow_color[gm0] = [1, 1, 1]
            pcd.colors = o3d.utility.Vector3dVector(flow_color)
        o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])


def vis_multiple(
    data_dir: str ="/home/kin/data/av2/h5py/demo/val",
    res_name: list = ["flow"],
    start_id: int = 0,
    point_size: float = 3.0,
    tone: str = 'dark',
    mode: str = "mul",
):
    if mode != "mul":
        return
    assert isinstance(res_name, list), "vis_multiple() needs a list as flow_mode"
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)
    o3d_vis = MyMultiVisualizer(view_file=VIEW_FILE, flow_mode=res_name)

    for v in o3d_vis.vis:
        opt = v.get_render_option()
        if tone == 'bright':
            background_color = np.asarray([216, 216, 216]) / 255.0  # offwhite
            # background_color = np.asarray([1, 1, 1])
            pcd_color = [0.25, 0.25, 0.25]
        elif tone == 'dark':
            background_color = np.asarray([80/255, 90/255, 110/255])  # dark
            pcd_color = [1., 1., 1.]

        opt.background_color = background_color
        opt.point_size = point_size

    data_id = start_id
    pbar = tqdm(range(0, len(dataset)))

    while data_id >= 0 and data_id < len(dataset):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0

        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

        pcd_list = []
        for mode in res_name:
            pcd = o3d.geometry.PointCloud()
            if mode in ['dufo', 'label']:
                labels = data[mode]
                pcd_i = o3d.geometry.PointCloud()
                for label_i in np.unique(labels):
                    pcd_i.points = o3d.utility.Vector3dVector(pc0[labels == label_i][:, :3])
                    if label_i <= 0:
                        pcd_i.paint_uniform_color([1.0, 1.0, 1.0])
                    else:
                        pcd_i.paint_uniform_color(color_map[label_i % len(color_map)])
                    pcd += pcd_i
            elif mode in data:
                pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
                flow = data[mode] - pose_flow # ego motion compensation here.
                flow_color = flow_to_rgb(flow) / 255.0
                is_dynamic = np.linalg.norm(flow, axis=1) > 0.1
                flow_color[~is_dynamic] = pcd_color
                flow_color[gm0] = pcd_color
                pcd.colors = o3d.utility.Vector3dVector(flow_color)
            pcd_list.append([pcd, create_bev_square(), 
                            create_bev_square(size=204.8, color=[195/255,86/255,89/255]), 
                            o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])
        o3d_vis.update(pcd_list)

        data_id += o3d_vis.playback_direction
        pbar.update(o3d_vis.playback_direction)


if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(check_flow)
    fire.Fire(vis)
    fire.Fire(vis_multiple)
    print(f"Time used: {time.time() - start_time:.2f} s")