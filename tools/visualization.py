"""
Scene Flow Visualization Tool
=============================
Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
Author: Qingwen Zhang (https://kin-zhang.github.io/), Ajinkya Khoche (https://ajinkyakhoche.github.io/)

Part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).

Usage (Fire class-based):
    # Visualize flow with color coding
    python tools/visualization.py vis --data_dir /path/to/data --res_name flow
    
    # Compare multiple results side-by-side
    python tools/visualization.py vis --data_dir /path/to/data --res_name "[flow, flow_est]"
    
    # Show flow as vector lines
    python tools/visualization.py vector --data_dir /path/to/data
    
    # Check flow with pc0, pc1, and flowed pc0
    python tools/visualization.py check --data_dir /path/to/data
    
    # Show error heatmap
    python tools/visualization.py error --data_dir /path/to/data --res_name "[raw, flow_est]"
    
Keys:
    [SPACE] play/pause    [D] next frame    [A] prev frame
    [P] save screenshot   [E] save error bar
    [S] sync viewpoint across windows (multi-window mode)
    [ESC/Q] quit
"""

import numpy as np
import fire
import time
from tqdm import tqdm
import open3d as o3d
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.utils.mics import flow_to_rgb, error_to_color
from src.utils import npcal_pose0to1
from src.utils.o3d_view import O3DVisualizer, color_map
from src.dataset import HDF5Dataset

VIEW_FILE = f"{BASE_DIR}/assets/view/demo.json"
NO_COLOR = [1, 1, 1]


def _ensure_list(val):
    """Ensure value is a list."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


class SceneFlowVisualizer:
    """
    Open3D-based Scene Flow Visualizer.
    
    Supports multiple visualization modes as class methods,
    compatible with python-fire for CLI usage.
    """
    
    def __init__(
        self,
        data_dir: str = "/home/kin/data/av2/preprocess/sensor/mini",
        res_name: str = "flow",
        start_id: int = 0,
        num_frames: int = 2,
        rgm: bool = True, # remove ground mask
        slc: bool = False, # show lidar centers
        point_size: float = 3.0,
        max_distance: float = 50.0,
        bg_color: tuple = (80/255, 90/255, 110/255),
    ):
        """
        Initialize the visualizer.
        
        Args:
            data_dir: Path to HDF5 dataset directory
            res_name: Result name(s) to visualize (string or list)
            start_id: Starting frame index
            point_size: Point size for rendering
            rgm: Remove ground mask if True
            slc: Show LiDAR sensor centers if True
            num_frames: Number of frames for history mode
            max_distance: Maximum distance filter for points
            bg_color: Background color as RGB tuple (0-1 range)
        """
        self.data_dir = data_dir
        self.res_names = _ensure_list(res_name)
        self.start_id = start_id
        self.point_size = point_size
        self.rgm = rgm
        self.num_frames = num_frames
        self.max_distance = max_distance
        self.bg_color = bg_color
        self.show_lidar_centers = slc
    
    def _load_dataset(self, vis_name=None, n_frames=2):
        """Load HDF5 dataset."""
        vis_name = vis_name or self.res_names
        return HDF5Dataset(self.data_dir, vis_name=vis_name, n_frames=n_frames, index_flow='flow' in vis_name)
    
    def _create_visualizer(self, res_name=None):
        """Create O3DVisualizer instance."""
        res_name = res_name or self.res_names
        return O3DVisualizer(
            view_file=VIEW_FILE,
            res_name=res_name,
            point_size=self.point_size,
            bg_color=self.bg_color,
        )
    
    def _filter_ground_and_distance(self, pc, gm):
        """Apply ground mask and distance filter."""
        if not self.rgm:
            gm = np.zeros_like(gm)
        distance = np.linalg.norm(pc[:, :3], axis=1)
        return gm | (distance > self.max_distance)
    
    def _compute_pose_flow(self, pc0, pose0, pose1):
        """Compute ego-motion flow."""
        ego_pose = npcal_pose0to1(pose0, pose1)
        return pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
    
    # -------------------------------------------------------------------------
    # Visualization Modes (Fire subcommands)
    # -------------------------------------------------------------------------
    
    def vis(self):
        """
        Visualize scene flow with color-coded dynamic motion.
        
        Supports single or multiple result names for side-by-side comparison.
        Colors represent flow direction (after ego-motion compensation).
        """
        dataset = self._load_dataset()
        o3d_vis = self._create_visualizer()
        
        data_id = self.start_id
        pbar = tqdm(range(self.start_id, len(dataset)))
        
        while 0 <= data_id < len(dataset):
            data = dataset[data_id]
            pbar.set_description(f"id: {data_id}, scene: {data['scene_id']}, ts: {data['timestamp']}")
            
            pc0 = data['pc0']
            gm0 = self._filter_ground_and_distance(pc0, data['gm0'])
            pose_flow = self._compute_pose_flow(pc0, data['pose0'], data['pose1'])
            
            if self.rgm:
                pc0 = pc0[~gm0]
                pose_flow = pose_flow[~gm0]
            
            pcd_list = []
            for single_res in self.res_names:
                pcd = o3d.geometry.PointCloud()
                
                # Instance/cluster visualization
                if single_res in ['dufo', 'cluster', 'dufocluster', 'flow_instance_id', 
                                  'ground_mask', 'pc0_dynamic'] and single_res in data:
                    labels = data[single_res][~gm0] if self.rgm else data[single_res]
                    pcd = self._color_by_labels(pc0, labels)
                
                # Flow visualization
                elif single_res in data:
                    pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
                    flow = (data[single_res][~gm0] if self.rgm else data[single_res]) - pose_flow
                    flow_color = flow_to_rgb(flow) / 255.0
                    is_dynamic = np.linalg.norm(flow, axis=1) > 0.08
                    flow_color[~is_dynamic] = NO_COLOR
                    if not self.rgm:
                        flow_color[gm0] = NO_COLOR
                    pcd.colors = o3d.utility.Vector3dVector(flow_color)
                
                # Raw point cloud
                elif single_res == 'raw':
                    pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
                
                pcd_list.append([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])
                
                # show lidar centers
                if self.show_lidar_centers and 'lidar_center' in data:
                    lidar_center = data['lidar_center']
                    for lidar_num in range(lidar_center.shape[0]):
                        pcd_list[-1].append(
                            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(
                                lidar_center[lidar_num]
                            )
                        )
                        
            o3d_vis.update(pcd_list, index=data_id)
            data_id += o3d_vis.playback_direction
            pbar.update(o3d_vis.playback_direction)
    
    def check(self):
        """
        Check flow by showing pc0 (red), pc1 (green), and pc0+flow (blue).
        
        Useful for verifying flow correctness.
        """
        res_name = self.res_names[0] if self.res_names else "flow"
        dataset = self._load_dataset(vis_name=res_name)
        o3d_vis = self._create_visualizer(res_name=res_name)
        
        data_id = self.start_id
        pbar = tqdm(range(self.start_id, len(dataset)))
        
        while 0 <= data_id < len(dataset):
            data = dataset[data_id]
            pbar.set_description(f"id: {data_id}, scene: {data['scene_id']}, ts: {data['timestamp']}")
            
            if res_name not in dataset[data_id]:
                print(f"'{res_name}' not in dataset, skipping id {data_id}")
                data_id += 1
                continue
            
            data = dataset[data_id]
            pbar.set_description(f"id: {data_id}, scene: {data['scene_id']}, ts: {data['timestamp']}")
            
            pc0, pc1 = data['pc0'], data['pc1']
            if self.rgm:
                pc0 = pc0[~data['gm0']]
                pc1 = pc1[~data['gm1']]
            
            # Red: pc0
            pcd0 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(pc0[:, :3])
            pcd0.paint_uniform_color([1.0, 0.0, 0.0])
            
            # Green: pc1
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3])
            pcd1.paint_uniform_color([0.0, 1.0, 0.0])
            
            # Blue: pc0 + flow
            res_flow = data[res_name][~data['gm0']] if self.rgm else data[res_name]
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + res_flow)
            pcd2.paint_uniform_color([0.0, 0.0, 1.0])
            
            o3d_vis.update([pcd0, pcd1, pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])
            data_id += o3d_vis.playback_direction
            pbar.update(o3d_vis.playback_direction)

    def vector(self):
        """
        Visualize flow as red vector lines from source to target.
        
        Shows pc0 (green), pc1 (blue), and flow vectors (red lines).
        """
        res_name = self.res_names[0] if self.res_names else "flow"
        dataset = self._load_dataset(vis_name=res_name)
        o3d_vis = O3DVisualizer(
            view_file=VIEW_FILE,
            res_name=res_name,
            point_size=self.point_size,
            bg_color=(1, 1, 1),  # White background for vector mode
        )
        
        data_id = self.start_id
        pbar = tqdm(range(self.start_id, len(dataset)))
        
        while 0 <= data_id < len(dataset):
            data = dataset[data_id]
            pbar.set_description(f"id: {data_id}, scene: {data['scene_id']}, ts: {data['timestamp']}")
            
            if res_name not in dataset[data_id]:
                print(f"'{res_name}' not in dataset, skipping id {data_id}")
                data_id += 1
                continue

            pc0 = data['pc0']
            gm0 = self._filter_ground_and_distance(pc0, data['gm0'])
            
            ego_pose = np.linalg.inv(data['pose1']) @ data['pose0']
            pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
            flow = data[res_name] - pose_flow
            
            # Green: pc0 transformed
            vis_pc = pc0[:, :3][~gm0] + pose_flow[~gm0]
            pcd0 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(vis_pc)
            pcd0.paint_uniform_color([0, 1, 0])
            
            # Blue: pc1
            pcd1 = o3d.geometry.PointCloud()
            gm1 = self._filter_ground_and_distance(data['pc1'], data['gm1'])
            pcd1.points = o3d.utility.Vector3dVector(data['pc1'][:, :3][~gm1])
            pcd1.paint_uniform_color([0.0, 0.0, 1])
            
            # Red: flow vectors
            line_set = self._create_flow_lines(vis_pc, flow[~gm0], color=[1, 0, 0])
            
            o3d_vis.update([pcd0, pcd1, line_set, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)],
                          index=data_id)
            data_id += o3d_vis.playback_direction
            pbar.update(o3d_vis.playback_direction)
            
    def error(self, max_error: float = 0.35):
        """
        Visualize flow error as heatmap (hot colormap).
        
        Args:
            max_error: Maximum error for color scaling (meters)
        """
        dataset = self._load_dataset()
        o3d_vis = self._create_visualizer()
        o3d_vis.bg_color = np.asarray([216, 216, 216]) / 255.0  # Off-white
        
        data_id = self.start_id
        pbar = tqdm(range(0, len(dataset)))
        
        while 0 <= data_id < len(dataset):
            data = dataset[data_id]
            pbar.set_description(f"id: {data_id}, scene: {data['scene_id']}, ts: {data['timestamp']}")
            
            pc0 = data['pc0']
            gm0 = self._filter_ground_and_distance(pc0, data['gm0'])
            
            gt_flow = data["flow"][~gm0] if self.rgm else data["flow"]
            if self.rgm:
                pc0 = pc0[~gm0]
            
            pcd_list = []
            for single_res in self.res_names:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
                
                res_flow = None
                if single_res in data:
                    res_flow = data[single_res][~gm0] if self.rgm else data[single_res]
                elif single_res == 'raw':
                    res_flow = self._compute_pose_flow(pc0, data['pose0'], data['pose1'])
                
                if res_flow is not None:
                    error_mag = np.linalg.norm(gt_flow - res_flow, axis=-1)
                    error_mag[error_mag < 0.05] = 0
                    error_color = error_to_color(error_mag, max_error=max_error, color_map="hot") / 255.0
                    if not self.rgm:
                        error_color[gm0] = NO_COLOR
                    pcd.colors = o3d.utility.Vector3dVector(error_color)
                    pcd_list.append([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])
            
            o3d_vis.update(pcd_list, index=data_id, value=max_error)
            data_id += o3d_vis.playback_direction
            pbar.update(o3d_vis.playback_direction)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _color_by_labels(self, pc, labels):
        """Create point cloud colored by instance labels."""
        pcd = o3d.geometry.PointCloud()
        for label_i in np.unique(labels):
            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(pc[labels == label_i][:, :3])
            if label_i <= 0:
                pcd_i.paint_uniform_color(NO_COLOR)
            else:
                pcd_i.paint_uniform_color(color_map[label_i % len(color_map)])
            pcd += pcd_i
        return pcd
    
    def _create_flow_lines(self, source_pts, flow, color=[1, 0, 0]):
        """Create line set for flow visualization."""
        line_set = o3d.geometry.LineSet()
        target_pts = source_pts + flow
        line_set_points = np.concatenate([source_pts, target_pts], axis=0)
        lines = np.array([[i, i + len(source_pts)] for i in range(len(source_pts))])
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        return line_set

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(SceneFlowVisualizer)
    print(f"Time used: {time.time() - start_time:.2f} s")