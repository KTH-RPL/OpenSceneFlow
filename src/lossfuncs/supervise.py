"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Define the supervised (needed GT) loss function for training.
#
"""
import torch
import numpy as np
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)
from src.utils.av2_eval import CATEGORY_TO_INDEX, BUCKETED_METACATAGORIES

# check: https://arxiv.org/abs/2508.17054
def deltaflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    instances = res_dict['gt_instance']
    
    reassign_meta = torch.zeros_like(classes, dtype=torch.int, device=classes.device)
    for i, cats in enumerate(BUCKETED_METACATAGORIES):
        selected_classes_ids = [CATEGORY_TO_INDEX[cat] for cat in BUCKETED_METACATAGORIES[cats]]
        reassign_meta[torch.isin(classes, torch.tensor(selected_classes_ids, device=classes.device))] = i

    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)
    speed = torch.linalg.vector_norm(gt, dim=-1) / 0.1
    
    weight_loss = deflowLoss(res_dict)['loss']

    classes_loss = 0.0
    weight = [0.1, 1.0, 2.0, 2.5, 1.5] # BACKGROUND, CAR, PEDESTRIAN, WHEELED, OTHER
    for class_id in range(len(BUCKETED_METACATAGORIES)):
        mask = reassign_meta == class_id
        for loss_ in [0.1 * pts_loss[(speed < 0.4) & mask].mean(), 
                      0.4 * pts_loss[(speed >= 0.4) & (speed <= 1.0) & mask].mean(), 
                      0.5 * pts_loss[(speed > 1.0) & mask].mean()]:
            classes_loss += torch.nan_to_num(loss_, nan=0.0) * weight[class_id]

    instance_loss, cnt = 0.0, 0
    if instances is not None:
        for instance_id in torch.unique(instances):
            mask = instances == instance_id
            reassign_meta_instance = reassign_meta[mask]
            class_id = torch.mode(reassign_meta_instance, 0).values.item()
            loss_ = pts_loss[mask].mean()
            if speed[mask].mean() <= 0.4:
                continue
            instance_loss += (loss_ * torch.exp(loss_) * weight[class_id])
            cnt += 1
        instance_loss /= (cnt if cnt > 0 else 1)
    return {'loss': weight_loss + classes_loss + instance_loss}

# check: https://arxiv.org/abs/2401.16122
def deflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']

    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    weight_loss = 0.0
    for loss_ in [pts_loss[speed < 0.4].mean(), 
                  pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean(), 
                  pts_loss[speed > 1.0].mean()]:
        weight_loss += torch.nan_to_num(loss_, nan=0.0)

    return {'loss': weight_loss}

# designed from MambaFlow: https://github.com/SCNU-RISLAB/MambaFlow
def mambaflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
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
    for loss_ in [pts_loss[speed < turning_speed].mean(), 
                  pts_loss[(speed >= turning_speed) & (speed <= 2)].mean(), 
                  pts_loss[speed > 2].mean()]:
        weight_loss += torch.nan_to_num(loss_, nan=0.0)
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
