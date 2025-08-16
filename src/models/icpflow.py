
"""
This file is from: https://github.com/yanconglin/ICP-Flow/blob/main/utils_flow.py
with slightly modification to have unified format with all benchmark.
"""

import dztimer, torch
import torch.nn as nn
import numpy as np
import hdbscan
import torch

from .basic import cal_pose0to1
from .basic.icpflow_lib import track

def flow_estimation(src_points, dst_points,
                    src_labels, dst_labels, 
                    pairs, transformations, pose):
    unqs = np.unique(src_labels.astype(int))
    # sanity check:  src labels include -1e8 and -1; segment labels start from 0.
    # -1e8: ground, 
    # -1: unmatched/unclustered segments in non-ground points
    # assert len(unqs)-2 == max(src_label)+1
    # print('unqs: ', len(unqs), unqs)

    assert len(src_points) == len(src_labels)
    flow = np.zeros((len(src_points), 3)) 
    for unq in unqs:
        idxs = src_labels==unq
        xyz = src_points[src_labels==unq, 0:3]
        if unq in pairs[:, 0]:
            idx = np.flatnonzero(unq==pairs[:,0])
            assert len(idx)==1
            idx = idx.item()
            transformation = transformations[idx]
        else:
            transformation = np.eye(4)

        flow_i = calculate_flow_rigid(xyz, transformation @ pose)
        flow[idxs] = flow_i

    return flow

def calculate_flow_rigid(points, transformation):
    points_tmp = transform_points(points, transformation)
    flow = points_tmp - points
    return flow

def transform_points(xyz, pose):
    assert xyz.shape[1]==3
    xyzh = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    xyzh_tmp = xyzh @ pose.T
    return xyzh_tmp[:, 0:3]

def cluster_pcd(points, min_cluster_size=20, num_clusters=200):

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=min_cluster_size, min_samples=None
                            )
    clusterer.fit(points[:, 0:3])
    labels = clusterer.labels_.copy()
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:num_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1 # unclustered points

    return labels

def cluster_labels(pose0, pose1, pc0_wo_ground, pc1_wo_ground):

    point_src = pc0_wo_ground
    point_dst = pc1_wo_ground
    
    pose = cal_pose0to1(pose0, pose1)
    point_src_ego = point_src @ pose[:3, :3].T + pose[:3, 3]
    points_tmp = np.concatenate([point_dst.cpu().numpy(), point_src_ego.cpu().numpy()], axis=0)

    label_tmp = cluster_pcd(points_tmp)
    label_src = label_tmp[len(point_dst):]
    label_dst = label_tmp[0:len(point_dst)]

    return point_src_ego, point_dst, label_src, label_dst, pose.cpu().numpy()

from dataclasses import dataclass
@dataclass
class dataargs:
    thres_dist: float = 0.1
    translation_frame: float = 1.67
    chunk_size: int = 50
    thres_iou: float = 0.2
    max_points: int = 10000
    min_cluster_size: int = 20
    thres_box: float = 0.1
    thres_rot: float = 0.1
    thres_error: float = 0.2
    speed: float = 1.67

class ICPFlow(nn.Module):
    def __init__(self, thres_dist: float = 0.1,
                 translation_frame: float = 1.67,
                 chunk_size: int = 50,
                 thres_iou: float = 0.2, 
                 max_points: int = 10000, 
                 min_cluster_size: int = 20, 
                 thres_box: float = 0.1,  
                 thres_rot: float = 0.1,
                 thres_error: float = 0.2,
                 speed: float = 1.67,
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3]
                 ):
        super().__init__()
        self.args = dataargs(thres_dist, translation_frame, chunk_size,
                             thres_iou, max_points, min_cluster_size, thres_box,
                             thres_rot, thres_error, speed)
        self.timer = dztimer.Timing()
        self.timer.start("NSFP Model Inference")
        self.point_cloud_range = point_cloud_range
        print(f"\n---LOG[model]: ICPFlow setup successfully, hyperparameters Settings: thres_dist={thres_dist}, min_cluster_size={min_cluster_size}, speed={speed}.")

    def range_limit_(self, pc):
        """
        Limit the point cloud to the given range.
        """
        mask = (pc[:, 0] >= self.point_cloud_range[0]) & (pc[:, 0] <= self.point_cloud_range[3]) & \
               (pc[:, 1] >= self.point_cloud_range[1]) & (pc[:, 1] <= self.point_cloud_range[4]) & \
               (pc[:, 2] >= self.point_cloud_range[2]) & (pc[:, 2] <= self.point_cloud_range[5])
        return pc[mask], mask
    
    def forward(self, batch):
        batch_sizes = len(batch["pose0"])
        device = batch['pose0'][0].device
        pose_flows = []
        batch_final_flow = []
        for batch_id in range(batch_sizes):
            self.timer[0].start("Data Processing")
            pc0 = batch["pc0"][batch_id]
            pc1 = batch["pc1"][batch_id]
            selected_pc0, rm0 = self.range_limit_(pc0)
            selected_pc1, rm1 = self.range_limit_(pc1)
            self.timer[0][0].start("pose")
            if 'ego_motion' in batch:
                pose_0to1 = batch['ego_motion'][batch_id]
            else:
                pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)

            self.timer[0].stop()

            point_src, point_dst, label_src, label_dst, pose = cluster_labels(batch["pose0"][batch_id], batch["pose1"][batch_id], selected_pc0, selected_pc1) 
            translation_frame = max(self.args.speed, np.linalg.norm(pose[0:3, -1])) * 2

            with torch.inference_mode(True):
                with torch.device(device):        
                    point_src = point_src.float().to(device)
                    point_dst = point_dst.float().to(device)
                    pairs, transformations = track(self.args, point_src, point_dst, torch.from_numpy(label_src).float().to(device), \
                                                        torch.from_numpy(label_dst).float().to(device))
            pairs = pairs.cpu().numpy()
            transformations = transformations.cpu().numpy()

            # assigning the same labels to corresponding instances; useful for visualization 
            # tracker_src, tracker_dst = trackers2labels(label_src, labels_dst[0], pairs)
            point_src = selected_pc0.cpu().numpy()
            point_dst = selected_pc1.cpu().numpy()

            flow = flow_estimation(src_points=point_src, dst_points=point_dst, 
                                src_labels=label_src, dst_labels=label_dst, 
                                pairs=pairs, transformations=transformations, pose=pose)
                
            final_flow = torch.zeros_like(pc0)
            final_flow[rm0] = torch.from_numpy(flow).float().to(device) - pose_flows[batch_id]
            batch_final_flow.append(final_flow)

        res_dict = {"flow": batch_final_flow,
                    "pose_flow": pose_flows
                    }
        
        return res_dict