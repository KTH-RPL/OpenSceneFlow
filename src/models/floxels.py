"""
# Created: 2024-07-27 11:40
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * TODO
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Re-implementation of Floxels (CVPR 2025): Fast Unsupervised Voxel Based Scene Flow Estimation.
#              The flow of one frame is a voxel grid, optimized against the frames before and after it.
# one more package need install: pip install FastGeodis --no-build-isolation --no-cache-dir
"""

import re
import dztimer, torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
import FastGeodis # extra package, the distance transform (DT) needs it

from .basic.opt_module import EarlyStopping, DT
from .basic import cal_pose0to1


class VoxelGrid(nn.Module):
    """The flow field: a 3D grid whose vertices store a flow vector, the flow of a point is the trilinear interpolation of them."""
    def __init__(self, point_cloud_range, voxel_size):
        super().__init__()
        self.register_buffer('grid_min', torch.tensor(list(point_cloud_range[:3]), dtype=torch.float32))
        self.voxel_size = voxel_size
        size = [int((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size) + 1 for i in range(3)]
        self.grid = nn.Parameter(torch.zeros(*size, 3))

    def forward(self, points):
        # continuous grid index of every point [N, 3]
        coords = (points - self.grid_min) / self.voxel_size
        x0, y0, z0 = [coords[:, i].floor().long().clamp(0, n - 1) for i, n in enumerate(self.grid.shape[:3])]
        x1, y1, z1 = [(coords[:, i].floor().long() + 1).clamp(0, n - 1) for i, n in enumerate(self.grid.shape[:3])]
        xd, yd, zd = [(coords[:, i] - low.float()).unsqueeze(1) for i, low in enumerate([x0, y0, z0])]

        grid = self.grid
        c00 = grid[x0, y0, z0] * (1 - xd) + grid[x1, y0, z0] * xd
        c01 = grid[x0, y0, z1] * (1 - xd) + grid[x1, y0, z1] * xd
        c10 = grid[x0, y1, z0] * (1 - xd) + grid[x1, y1, z0] * xd
        c11 = grid[x0, y1, z1] * (1 - xd) + grid[x1, y1, z1] * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        return c0 * (1 - zd) + c1 * zd


def cluster(pc, min_samples=4, eps=0.5):
    """DBSCAN cluster id of every point, -1 for the points without a cluster."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pc.detach().cpu().numpy()).labels_
    return torch.tensor(labels, dtype=torch.int, device=pc.device)


def cluster_loss(flow, clusters):
    """Distance of every flow to the mean flow of its cluster: the points of one cluster should move together."""
    clusters_unique, indices = torch.unique(clusters, return_inverse=True)
    cluster_sums = torch.zeros((clusters_unique.shape[0], 3), device=flow.device)
    cluster_counts = torch.zeros((clusters_unique.shape[0], 1), device=flow.device)
    cluster_sums.scatter_add_(0, indices[:, None].expand(-1, 3), flow)
    cluster_counts.scatter_add_(0, indices[:, None], torch.ones_like(flow[:, :1]))
    cluster_means = cluster_sums / cluster_counts
    return (flow - cluster_means[indices]).norm(dim=1)


class Floxels(nn.Module):
    # default parameters from Sec 3.1
    def __init__(self, grid_factor=10., itr_num=500, lr=0.05, min_delta=0.01, early_patience=250,
                 verbose=False, point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3], voxel_size=0.5,
                 cluster_weight=0.6):
        super().__init__()
        self.grid_factor = grid_factor # grid cell size of the distance transform = 1/grid_factor.
        self.iteration_num = itr_num
        self.min_delta = min_delta
        self.lr = lr
        self.early_patience = early_patience
        self.verbose = verbose
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.cluster_weight = cluster_weight

        self.timer = dztimer.Timing()
        self.timer.start("Floxels Model Inference")
        print(f"\n---LOG [model]: Floxels setup itr_num: {itr_num}, lr: {lr}, early_patience: {early_patience} with min_delta: {min_delta}.")

    def optimize(self, pc0, other_frames):
        """
        pc0: [N, 3] the points to estimate the flow for.
        other_frames: {k: [M, 3] points of the frame k steps after pc0}, k = ..., -2, -1, 1, 2, ...
        All frames are in the same coordinate. Returns the flow of pc0 to the next frame (k=1) with the lowest loss.
        """
        device = pc0.device
        net = VoxelGrid(self.point_cloud_range, self.voxel_size).to(device)

        self.timer[2].start("DT Map Building")
        dts = {}
        pc0_min, pc0_max = pc0.min(0)[0], pc0.max(0)[0]
        for k, points in other_frames.items():
            grid_min = torch.floor(torch.minimum(pc0_min, points.min(0)[0]) * self.grid_factor - 1) / self.grid_factor
            grid_max = torch.ceil(torch.maximum(pc0_max, points.max(0)[0]) * self.grid_factor + 1) / self.grid_factor
            dts[k] = DT(points, grid_min, grid_max, self.grid_factor, device)
        self.timer[2].stop()

        self.timer[3].start("Clustering")
        clusters = cluster(pc0)
        self.timer[3].stop()

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
        early_stopping = EarlyStopping(patience=self.early_patience, min_delta=self.min_delta)
        best = {'loss': torch.inf}

        self.timer[4].start("Optimization")
        for itr in range(self.iteration_num):
            optimizer.zero_grad()
            flow = net(pc0)

            # constant velocity: k frames later the points of pc0 are at pc0 + k * flow. A frame further away counts less.
            loss = 0
            for k, dt in dts.items():
                loss += dt.torch_bilinear_distance(pc0 + k * flow, truncate_dist=5.0).mean() / k**2
            # NOTE: the weight of the distance loss (lambda_d) is 1, confirmed with Hanqiu

            loss += self.cluster_weight * len(dts) * cluster_loss(flow[clusters >= 0], clusters[clusters >= 0]).mean()
            # flow norm regularizer, its weight goes from 0.1 to 0.01 in the first 100 iterations
            loss += np.interp(itr, [0, 100], [0.1, 0.01]) * len(dts) * flow.norm(p=2, dim=1).mean()

            if loss <= best['loss']:
                best = {'loss': loss.item(), 'flow': flow.detach()}
            if early_stopping.step(loss) and 'flow' in best: # at least one step
                break

            loss.backward()
            optimizer.step()
            scheduler.step()
        self.timer[4].stop()

        if self.verbose:
            self.timer.print(random_colors=True, bold=True)
        return best

    def in_range(self, pc):
        pc_range = pc.new_tensor(list(self.point_cloud_range))
        return ((pc >= pc_range[:3]) & (pc <= pc_range[3:])).all(dim=1)

    def forward(self, batch):
        batch_flow, batch_loss = [], []
        for batch_id in range(len(batch["pose0"])):
            self.timer[0].start("Data Preprocess")
            # Every frame the dataloader gives, by its number of steps after pc0: pch2 -> -2, pch1 -> -1, pc0 -> 0, pc1 -> 1, pc2 -> 2, ...
            # They are all moved into the coordinate of pc1 (ego-motion compensation), so the grid only has the flow of the moving objects.
            frames = {}
            for key in batch:
                match = re.fullmatch(r'pc(h?)(\d+)', key)
                if match is None:
                    continue
                k = -int(match.group(2)) if match.group(1) else int(match.group(2))
                pose_to_pc1 = cal_pose0to1(batch[key.replace('pc', 'pose')][batch_id], batch["pose1"][batch_id])
                frames[k] = batch[key][batch_id] @ pose_to_pc1[:3, :3].T + pose_to_pc1[:3, 3]
            pc0 = frames.pop(0)
            pc0_in_range = self.in_range(pc0)
            self.timer[0].stop()

            # since the runner disables the gradient in eval.
            with torch.inference_mode(False):
                with torch.enable_grad():
                    other_frames = {k: frames[k][self.in_range(frames[k])].clone() for k in sorted(frames)}
                    best = self.optimize(pc0[pc0_in_range].clone(), other_frames)

            flow = torch.zeros_like(pc0)
            flow[pc0_in_range] = best['flow']
            batch_flow.append(flow)
            batch_loss.append(best['loss'])

        return {"flow": batch_flow, "loss": batch_loss}
