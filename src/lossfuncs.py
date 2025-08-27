"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow) and SeFlow (https://github.com/KTH-RPL/SeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Define the loss function for training.
"""
import torch
import numpy as np
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()
from src.utils.av2_eval import CATEGORY_TO_INDEX, BUCKETED_METACATAGORIES

# NOTE(Qingwen 24/07/06): squared, so it's sqrt(4) = 2m, in 10Hz the vel = 20m/s ~ 72km/h
# If your scenario is different, may need adjust this TRUNCATED to 80-120km/h vel.
TRUNCATED_DIST = 4

# ---------------------- Self-Supervised Flow Loss without GT ----------------------
def seflowLoss(res_dict, timer=None):
    pc0_label = res_dict['pc0_labels']
    pc1_label = res_dict['pc1_labels']

    pc0 = res_dict['pc0']
    pc1 = res_dict['pc1']

    est_flow = res_dict['est_flow']

    pseudo_pc1from0 = pc0 + est_flow

    unique_labels = torch.unique(pc0_label)
    pc0_dynamic = pc0[pc0_label > 0]
    pc1_dynamic = pc1[pc1_label > 0]
    # fpc1_dynamic = pseudo_pc1from0[pc0_label > 0]
    # NOTE(Qingwen): since we set THREADS_PER_BLOCK is 256
    have_dynamic_cluster = (pc0_dynamic.shape[0] > 256) & (pc1_dynamic.shape[0] > 256)

    # first item loss: chamfer distance
    # timer[5][1].start("MyCUDAChamferDis")
    # raw: pc0 to pc1, est: pseudo_pc1from0 to pc1, idx means the nearest index
    est_dist0, est_dist1, _, _ = MyCUDAChamferDis.disid_res(pseudo_pc1from0, pc1)
    raw_dist0, raw_dist1, raw_idx0, _ = MyCUDAChamferDis.disid_res(pc0, pc1)
    chamfer_dis = torch.mean(est_dist0[est_dist0 <= TRUNCATED_DIST]) + torch.mean(est_dist1[est_dist1 <= TRUNCATED_DIST])
    # timer[5][1].stop()
    
    # second item loss: dynamic chamfer distance
    # timer[5][2].start("DynamicChamferDistance")
    dynamic_chamfer_dis = torch.tensor(0.0, device=est_flow.device)
    if have_dynamic_cluster:
        dynamic_chamfer_dis += MyCUDAChamferDis(pseudo_pc1from0[pc0_label>0], pc1_dynamic, truncate_dist=TRUNCATED_DIST)
    # timer[5][2].stop()

    # third item loss: exclude static points' flow
    # NOTE(Qingwen): add in the later part on label==0
    static_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    
    # fourth item loss: same label points' flow should be the same
    # timer[5][3].start("SameClusterLoss")
    moved_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    moved_cluster_norms = torch.tensor([], device=est_flow.device)
    for label in unique_labels:
        mask = pc0_label == label
        if label == 0:
            # Eq. 6 in the paper
            static_cluster_loss += torch.linalg.vector_norm(est_flow[mask, :], dim=-1).mean()
        elif label > 0 and have_dynamic_cluster:
            cluster_id_flow = est_flow[mask, :]
            cluster_nnd = raw_dist0[mask]
            if cluster_nnd.shape[0] <= 0:
                continue

            # Eq. 8 in the paper
            sorted_idxs = torch.argsort(cluster_nnd, descending=True)
            nearby_label = pc1_label[raw_idx0[mask][sorted_idxs]] # nonzero means dynamic in label
            non_zero_valid_indices = torch.nonzero(nearby_label > 0)
            if non_zero_valid_indices.shape[0] <= 0:
                continue
            max_idx = sorted_idxs[non_zero_valid_indices.squeeze(1)[0]]
            
            # Eq. 9 in the paper
            max_flow = pc1[raw_idx0[mask][max_idx]] - pc0[mask][max_idx]

            # Eq. 10 in the paper
            moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_id_flow - max_flow), dim=-1)))
    
    if moved_cluster_norms.shape[0] > 0:
        moved_cluster_loss = moved_cluster_norms.mean() # Eq. 11 in the paper
    elif have_dynamic_cluster:
        moved_cluster_loss = torch.mean(raw_dist0[raw_dist0 <= TRUNCATED_DIST]) + torch.mean(raw_dist1[raw_dist1 <= TRUNCATED_DIST])
    # timer[5][3].stop()

    res_loss = {
        'chamfer_dis': chamfer_dis,
        'dynamic_chamfer_dis': dynamic_chamfer_dis,
        'static_flow_loss': static_cluster_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }
    return res_loss

# ---------------------- Supervised Flow Loss with GT ----------------------
# designed from MambaFlow: https://github.com/SCNU-RISLAB/MambaFlow
def mambaflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())

    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    velocities = speed.cpu().numpy()

    # 计算直方图，返回每个区间的计数和区间边界
    counts, bin_edges = np.histogram(velocities, bins=100, density=False)

    # 计算每个区间的点数占总点数的比例
    total_points = len(velocities)
    proportions = counts / total_points

    # 计算每个区间的中心位置，用于绘图
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 设置占比阈值
    proportion_threshold = 0.01  # 可以根据需要调整这个值

    # 找出第一个占比小于阈值的柱子
    first_below_threshold = next((i for i, prop in enumerate(proportions) if prop < proportion_threshold), None)
    turning_speed = bin_centers[first_below_threshold]

    weight_loss = 0.0
    speed_mid = 2
    speed_0 = pts_loss[speed < turning_speed].mean()
    speed_1 = pts_loss[(speed >= turning_speed) & (speed <= speed_mid)].mean()
    speed_2 = pts_loss[speed > speed_mid].mean()
    if ~speed_1.isnan():
        weight_loss += speed_1
    if ~speed_0.isnan():
        weight_loss += speed_0
    if ~speed_2.isnan():
        weight_loss += speed_2
    return {'loss': weight_loss}

def deflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']

    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    weight_loss = 0.0
    speed_0_4 = pts_loss[speed < 0.4].mean()
    speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
    speed_1_0 = pts_loss[speed > 1.0].mean()
    if ~speed_1_0.isnan():
        weight_loss += speed_1_0
    if ~speed_0_4.isnan():
        weight_loss += speed_0_4
    if ~speed_mid.isnan():
        weight_loss += speed_mid
    return {'loss': weight_loss}

# ref from zeroflow loss class FastFlow3DDistillationLoss()
def zeroflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    # gt_speed = torch.norm(gt, dim=1, p=2) * 10.0
    gt_speed = torch.linalg.vector_norm(gt, dim=-1) * 10.0
    
    mins = torch.ones_like(gt_speed) * 0.1
    maxs = torch.ones_like(gt_speed)
    importance_scale = torch.max(mins, torch.min(1.8 * gt_speed - 0.8, maxs))
    # error = torch.norm(pred - gt, dim=1, p=2) * importance_scale
    error = error * importance_scale
    return {'loss': error.mean()}

# ref from zeroflow loss class FastFlow3DSupervisedLoss()
def ff3dLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    # error = torch.norm(pred - gt, dim=1, p=2)
    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    is_foreground_class = (classes > 0) # 0 is background, ref: FOREGROUND_BACKGROUND_BREAKDOWN
    background_scalar = is_foreground_class.float() * 0.9 + 0.1
    error = error * background_scalar
    return {'loss': error.mean()}
