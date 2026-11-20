"""
# Created: 2026-11-20 16:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), AI Assistant (Claude Code)
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.
#
# Description: What eval.py and save.py do with one frame around the model:
#                  remove ground -> model -> flow on all points -> metrics / write to h5 / write leaderboard file.
#              Shared by src/trainer.py (feed-forward models) and src/runner.py (optimization-based models).
"""

import h5py, torch
import numpy as np

from .models.basic import cal_pose0to1
from .utils.av2_eval import write_output_file
from .utils.eval_metric import evaluate_leaderboard, evaluate_leaderboard_v2, evaluate_ssf


def run_model_wo_ground(model, batch):
    """Run the model on one frame from the eval/save dataloader: batch size 1, all points with ground.

    The model only takes non-ground points. The full pc0 is kept in batch['origin_pc0'].
    """
    batch['origin_pc0'] = batch['pc0'].clone()
    for key in [k for k in batch if k.startswith('pc') and f'gm{k[2:]}' in batch]: # pc0, pc1, pch1, ...
        batch[key] = batch[key][~batch[f'gm{key[2:]}']].unsqueeze(0)

    model.timer[12].start("One Scan")
    res_dict = model(batch)
    model.timer[12].stop()

    # NOTE (Qingwen): Since val and test, we will force set batch_size = 1
    batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
    res_dict = {key: res_dict[key][0] for key in res_dict if (res_dict[key]!=None and len(res_dict[key])>0) }
    return batch, res_dict


def flow_on_all_points(batch, res_dict):
    """Flow of every pc0 point, from the model result which only has the non-ground points.

    Returns (final_flow, pose_flow): the full flow, and the ego-motion part of it.
    Ground points and the points the model dropped only have the ego-motion flow.
    """
    pc0 = batch['origin_pc0']
    pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
    transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
    pose_flow = transform_pc0 - pc0

    non_ground = ~batch['gm0']
    if 'pc0_valid_point_idxes' in res_dict: # voxel-based models drop the points out of range
        est_flow = torch.zeros_like(pose_flow[non_ground])
        est_flow[res_dict['pc0_valid_point_idxes']] = res_dict['flow']
    else:
        est_flow = res_dict['flow']

    final_flow = pose_flow.clone()
    final_flow[non_ground] += est_flow
    return final_flow, pose_flow


def eval_frame(batch, final_flow, pose_flow):
    """Metrics of one frame on the points inside eval_mask.

    Returns (v1 three-way EPE, v2 bucketed, ssf range-wise), the three inputs of OfficialMetrics.step().
    """
    eval_mask = batch['eval_mask'].squeeze()
    args = (final_flow[eval_mask], pose_flow[eval_mask], batch['origin_pc0'][eval_mask], batch['flow'][eval_mask],
            batch['flow_is_valid'][eval_mask], batch['flow_category_indices'][eval_mask])
    return evaluate_leaderboard(*args), evaluate_leaderboard_v2(*args), evaluate_ssf(*args)


def write_flow_to_h5(h5_path, timestamp, res_name, flow):
    """Write the flow of one frame into the dataset h5 file as `res_name`, tools/visualization.py reads it."""
    key = str(timestamp)
    with h5py.File(h5_path, 'r+') as f:
        if res_name in f[key]:
            del f[key][res_name]
        f[key].create_dataset(res_name, data=flow.cpu().detach().numpy().astype(np.float32))


def write_leaderboard_file(batch, final_flow, pose_flow, save_res_path, leaderboard_version):
    """Write the result file of one test frame, to submit to the online leaderboard."""
    eval_mask = batch['eval_mask'].squeeze()
    save_pred_flow = final_flow[eval_mask, :3].cpu().detach().numpy()
    rigid_flow = pose_flow[eval_mask, :3].cpu().detach().numpy()
    is_dynamic = np.linalg.norm(save_pred_flow - rigid_flow, axis=1, ord=2) >= 0.05
    if leaderboard_version in [2, 3]:
        save_pred_flow = (final_flow - pose_flow).cpu().detach().numpy() # all points here... since 2rd version we need to save the relative flow.
    sweep_uuid = (batch['scene_id'], batch['timestamp'])
    write_output_file(save_pred_flow, is_dynamic, sweep_uuid, save_res_path, leaderboard_version=leaderboard_version)
