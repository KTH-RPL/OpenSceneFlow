"""
# 
# Created: 2024-04-14 11:57
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
#
# Reference to official evaluation scripts:
# - EPE Threeway: https://github.com/argoverse/av2-api/blob/main/src/av2/evaluation/scene_flow/eval.py
# - Bucketed EPE: https://github.com/kylevedder/BucketedSceneFlowEval/blob/master/bucketed_scene_flow_eval/eval/bucketed_epe.py
"""

import torch
import os, sys
import numpy as np
from typing import Dict, Final, List, Tuple
from tabulate import tabulate

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)
from src.utils.av2_eval import compute_metrics, compute_bucketed_epe, CLOSE_DISTANCE_THRESHOLD


# EPE Three-way: Foreground Dynamic, Background Dynamic, Background Static
# leaderboard link: https://eval.ai/web/challenges/challenge-page/2010/evaluation
def evaluate_leaderboard(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    gt_is_dynamic = torch.linalg.vector_norm(gt_flow - rigid_flow, dim=-1) >= 0.05
    mask_ = ~est_flow.isnan().any(dim=1) & ~rigid_flow.isnan().any(dim=1) & ~pc0[:, :3].isnan().any(dim=1) & ~gt_flow.isnan().any(dim=1)
    mask_no_nan = mask_ & ~gt_is_dynamic.isnan() & ~is_valid.isnan() & ~pts_ids.isnan()
    est_flow = est_flow[mask_no_nan, :]
    rigid_flow = rigid_flow[mask_no_nan, :]
    pc0 = pc0[mask_no_nan, :]
    gt_flow = gt_flow[mask_no_nan, :]
    gt_is_dynamic = gt_is_dynamic[mask_no_nan]
    is_valid = is_valid[mask_no_nan]
    pts_ids = pts_ids[mask_no_nan]

    est_is_dynamic = torch.linalg.vector_norm(est_flow - rigid_flow, dim=-1) >= 0.05
    is_close = torch.all(torch.abs(pc0[:, :2]) <= CLOSE_DISTANCE_THRESHOLD, dim=1)
    res_dict = compute_metrics(
        est_flow.detach().cpu().numpy().astype(float),
        est_is_dynamic.detach().cpu().numpy().astype(bool),
        gt_flow.detach().cpu().numpy().astype(float),
        pts_ids.detach().cpu().numpy().astype(np.uint8),
        gt_is_dynamic.detach().cpu().numpy().astype(bool),
        is_close.detach().cpu().numpy().astype(bool),
        is_valid.detach().cpu().numpy().astype(bool)
    )
    return res_dict

# EPE Bucketed: BACKGROUND, CAR, PEDESTRIAN, WHEELED_VRU, OTHER_VEHICLES
def evaluate_leaderboard_v2(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    # in x,y dis, ref to official evaluation: eval/base_per_frame_sceneflow_eval.py#L118-L119
    pc_distance = torch.linalg.vector_norm(pc0[:, :2], dim=-1)
    distance_mask = pc_distance <= CLOSE_DISTANCE_THRESHOLD

    mask_flow_non_nan = ~est_flow.isnan().any(dim=1) & ~rigid_flow.isnan().any(dim=1) & ~pc0[:, :3].isnan().any(dim=1) & ~gt_flow.isnan().any(dim=1)
    mask_eval = mask_flow_non_nan & ~is_valid.isnan() & ~pts_ids.isnan() & distance_mask
    rigid_flow = rigid_flow[mask_eval, :]
    est_flow = est_flow[mask_eval, :] - rigid_flow
    gt_flow = gt_flow[mask_eval, :] - rigid_flow # in v2 evaluation, we don't add rigid flow to evaluate
    is_valid = is_valid[mask_eval]
    pts_ids = pts_ids[mask_eval]

    res_dict = compute_bucketed_epe(
        est_flow.detach().cpu().numpy().astype(float),
        gt_flow.detach().cpu().numpy().astype(float),
        pts_ids.detach().cpu().numpy().astype(np.uint8),
        is_valid.detach().cpu().numpy().astype(bool),
    )
    return res_dict

# reference to official evaluation: bucketed_scene_flow_eval/eval/bucketed_epe.py
# python >= 3.7
from dataclasses import dataclass
import warnings
@dataclass(frozen=True, eq=True, repr=True)
class OverallError:
    static_epe: float
    dynamic_error: float

    def __repr__(self) -> str:
        static_epe_val_str = (
            f"{self.static_epe:0.6f}" if np.isfinite(self.static_epe) else f"{self.static_epe}"
        )
        dynamic_error_val_str = (
            f"{self.dynamic_error:0.6f}"
            if np.isfinite(self.dynamic_error)
            else f"{self.dynamic_error}"
        )
        return f"({static_epe_val_str}, {dynamic_error_val_str})"

    def to_tuple(self) -> Tuple[float, float]:
        return (self.static_epe, self.dynamic_error)

class BucketResultMatrix:
    def __init__(self, class_names: List[str], speed_buckets: List[Tuple[float, float]]):
        self.class_names = class_names
        self.speed_buckets = speed_buckets

        assert (
            len(self.class_names) > 0
        ), f"class_names must have at least one entry, got {len(self.class_names)}"
        assert (
            len(self.speed_buckets) > 0
        ), f"speed_buckets must have at least one entry, got {len(self.speed_buckets)}"

        # By default, NaNs are not counted in np.nanmean
        self.epe_storage_matrix = np.zeros((len(class_names), len(self.speed_buckets))) * np.nan
        self.speed_storage_matrix = np.zeros((len(class_names), len(self.speed_buckets))) * np.nan
        self.count_storage_matrix = np.zeros(
            (len(class_names), len(self.speed_buckets)), dtype=np.int64
        )

    def accumulate_value(
        self,
        class_name: str,
        speed_bucket: Tuple[float, float],
        average_epe: float,
        average_speed: float,
        count: int,
    ):
        assert count > 0, f"count must be greater than 0, got {count}"
        assert np.isfinite(average_epe), f"average_epe must be finite, got {average_epe}"
        assert np.isfinite(average_speed), f"average_speed must be finite, got {average_speed}"

        class_idx = self.class_names.index(class_name)
        speed_bucket_idx = self.speed_buckets.index(speed_bucket)

        prior_epe = self.epe_storage_matrix[class_idx, speed_bucket_idx]
        prior_speed = self.speed_storage_matrix[class_idx, speed_bucket_idx]
        prior_count = self.count_storage_matrix[class_idx, speed_bucket_idx]

        if np.isnan(prior_epe):
            self.epe_storage_matrix[class_idx, speed_bucket_idx] = average_epe
            self.speed_storage_matrix[class_idx, speed_bucket_idx] = average_speed
            self.count_storage_matrix[class_idx, speed_bucket_idx] = count
            return

        # Accumulate the average EPE and speed, weighted by the number of samples using np.mean
        self.epe_storage_matrix[class_idx, speed_bucket_idx] = np.average(
            [prior_epe, average_epe], weights=[prior_count, count]
        )
        self.speed_storage_matrix[class_idx, speed_bucket_idx] = np.average(
            [prior_speed, average_speed], weights=[prior_count, count]
        )
        self.count_storage_matrix[class_idx, speed_bucket_idx] += count

    def get_normalized_error_matrix(self) -> np.ndarray:
        error_matrix = self.epe_storage_matrix.copy()
        # For the 1: columns, normalize EPE entries by the speed
        error_matrix[:, 1:] = error_matrix[:, 1:] / self.speed_storage_matrix[:, 1:]
        return error_matrix

    def get_overall_class_errors(self, normalized: bool = True):
        #  -> dict[str, OverallError]
        if normalized:
            error_matrix = self.get_normalized_error_matrix()
        else:
            error_matrix = self.epe_storage_matrix.copy()
        static_epes = error_matrix[:, 0]
        # Hide the warning about mean of empty slice
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dynamic_errors = np.nanmean(error_matrix[:, 1:], axis=1)

        return {
            class_name: OverallError(static_epe, dynamic_error)
            for class_name, static_epe, dynamic_error in zip(
                self.class_names, static_epes, dynamic_errors
            )
        }

    def get_class_entries(self, class_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_idx = self.class_names.index(class_name)

        epe = self.epe_storage_matrix[class_idx, :]
        speed = self.speed_storage_matrix[class_idx, :]
        count = self.count_storage_matrix[class_idx, :]
        return epe, speed, count

    def get_mean_average_values(self, normalized: bool = True) -> OverallError:
        overall_errors = self.get_overall_class_errors(normalized=normalized)

        average_static_epe = np.nanmean([v.static_epe for v in overall_errors.values()])
        average_dynamic_error = np.nanmean([v.dynamic_error for v in overall_errors.values()])

        return OverallError(average_static_epe, average_dynamic_error)

class OfficialMetrics:
    def __init__(self):
        # same with BUCKETED_METACATAGORIES
        self.bucketed= {
            'BACKGROUND': {'Static': [], 'Dynamic': []},
            'CAR': {'Static': [], 'Dynamic': []},
            'OTHER_VEHICLES': {'Static': [], 'Dynamic': []},
            'PEDESTRIAN': {'Static': [], 'Dynamic': []},
            'WHEELED_VRU': {'Static': [], 'Dynamic': []},
            'Mean': {'Static': [], 'Dynamic': []}
        }

        self.epe_3way = {
            'EPE_FD': [],
            'EPE_BS': [],
            'EPE_FS': [],
            'IoU': [],
            'Three-way': []
        }

        self.norm_flag = False


        # bucket_max_speed, num_buckets, distance_thresholds set is from: eval/bucketed_epe.py#L226
        bucket_edges = np.concatenate([np.linspace(0, 2.0, 51), [np.inf]])
        speed_thresholds = list(zip(bucket_edges, bucket_edges[1:]))
        self.bucketedMatrix = BucketResultMatrix(
            class_names=['BACKGROUND', 'CAR', 'OTHER_VEHICLES', 'PEDESTRIAN', 'WHEELED_VRU'],
            speed_buckets=speed_thresholds
        )
    def step(self, epe_dict, bucket_dict):
        """
        This step function is used to store the results of **each frame**.
        """
        for key in epe_dict:
            self.epe_3way[key].append(epe_dict[key])

        for item_ in bucket_dict:
            category_name = item_.name
            speed_tuple = item_.speed_thresholds
            self.bucketedMatrix.accumulate_value(
                category_name,
                speed_tuple,
                item_.avg_epe,
                item_.avg_speed,
                item_.count,
            )

    def normalize(self):
        """
        This normalize mean average results between **frame and frame**.
        """
        for key in self.epe_3way:
            self.epe_3way[key] = np.mean(self.epe_3way[key])
        self.epe_3way['Three-way'] = np.mean([self.epe_3way['EPE_FD'], self.epe_3way['EPE_BS'], self.epe_3way['EPE_FS']])

        mean = self.bucketedMatrix.get_mean_average_values(normalized=True).to_tuple()
        class_errors = self.bucketedMatrix.get_overall_class_errors(normalized=True)
        for key in self.bucketed:
            if key == 'Mean':
                self.bucketed[key]['Static'] = mean[0]
                self.bucketed[key]['Dynamic'] = mean[1]
                continue
            for i, sub_key in enumerate(self.bucketed[key]):
                self.bucketed[key][sub_key] = class_errors[key].to_tuple()[i] # 0: static, 1: dynamic
        self.norm_flag = True
    
    def print(self):
        if not self.norm_flag:
            self.normalize()
        printed_data = []
        for key in self.epe_3way:
            printed_data.append([key,self.epe_3way[key]])
        print("Version 1 Metric on EPE Three-way:")
        print(tabulate(printed_data), "\n")

        printed_data = []
        for key in self.bucketed:
            printed_data.append([key, self.bucketed[key]['Static'], self.bucketed[key]['Dynamic']])
        print("Version 2 Metric on Category-based:")
        print(tabulate(printed_data, headers=["Class", "Static", "Dynamic"], tablefmt='orgtbl'), "\n")
    