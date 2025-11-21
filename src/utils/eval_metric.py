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
from typing import List, Tuple
from tabulate import tabulate

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)
from src.utils.av2_eval import compute_metrics, compute_bucketed_epe, compute_ssf_metrics, CLOSE_DISTANCE_THRESHOLD
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# EPE Three-way: Foreground Dynamic, Background Dynamic, Background Static
# leaderboard link: https://eval.ai/web/challenges/challenge-page/2010/evaluation
def evaluate_leaderboard(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    gt_is_dynamic = torch.linalg.vector_norm(gt_flow - rigid_flow, dim=-1) >= 0.05
    mask_ = ~est_flow.isnan().any(dim=1) & ~rigid_flow.isnan().any(dim=1) & ~pc0[:, :3].isnan().any(dim=1) & ~gt_flow.isnan().any(dim=1)
    # mask_no_nan = mask_ & ~gt_is_dynamic.isnan() & ~is_valid.isnan() & ~pts_ids.isnan()

    # added distance mask for v2 evaluation, 70x70 = 35m range for close distance
    pc_distance = torch.linalg.vector_norm(pc0[:, :2], dim=-1)
    distance_mask = pc_distance <= 35.0 #50.0 # No.... ~I remembered is_valid also limit the range to 50m~
    
    mask_eval = mask_ & ~gt_is_dynamic.isnan() & ~is_valid.isnan() & ~pts_ids.isnan() & distance_mask

    est_flow = est_flow[mask_eval, :]
    rigid_flow = rigid_flow[mask_eval, :]
    pc0 = pc0[mask_eval, :]
    gt_flow = gt_flow[mask_eval, :]
    gt_is_dynamic = gt_is_dynamic[mask_eval]
    is_valid = is_valid[mask_eval]
    pts_ids = pts_ids[mask_eval]

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

# EPE Range-wise: for SSF project.
def evaluate_ssf(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    # is_valid here will filter out the ground points.
    pc_distance = torch.linalg.vector_norm(pc0[:, :3], dim=-1)
    mask_flow_non_nan = ~est_flow.isnan().any(dim=1) & ~rigid_flow.isnan().any(dim=1) & ~pc0[:, :3].isnan().any(dim=1) & ~gt_flow.isnan().any(dim=1)
    mask_eval = mask_flow_non_nan & ~is_valid.isnan() & ~pts_ids.isnan()
    rigid_flow = rigid_flow[mask_eval, :]

    # NOTE(Qingwen): no pose flow (ego motion) in v2 and ssf evaluation, we focus on other agent's flow.
    est_flow = est_flow[mask_eval, :] - rigid_flow
    # NOTE(Ajinkya): set est_flow to zero (uncomment line below) to evaluate ego motion only.
    # # est_flow = torch.zeros_like(est_flow).to(est_flow.device)
    gt_flow = gt_flow[mask_eval, :] - rigid_flow 
    is_valid = is_valid[mask_eval]
    pc_distance = pc_distance[mask_eval]
    pts_ids = pts_ids[mask_eval]

    res_dict = compute_ssf_metrics(
        pc_distance.detach().cpu().numpy().astype(float),
        est_flow.detach().cpu().numpy().astype(float),
        gt_flow.detach().cpu().numpy().astype(float),
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
    def __init__(self, class_names: List[str], range_buckets: List[Tuple[float, float]]):
        self.class_names = class_names
        self.range_buckets = range_buckets

        assert (
            len(self.class_names) > 0
        ), f"class_names must have at least one entry, got {len(self.class_names)}"
        assert (
            len(self.range_buckets) > 0
        ), f"range_buckets must have at least one entry, got {len(self.range_buckets)}"

        # By default, NaNs are not counted in np.nanmean
        self.epe_storage_matrix = np.zeros((len(class_names), len(self.range_buckets))) * np.NaN
        self.range_storage_matrix = np.zeros((len(class_names), len(self.range_buckets))) * np.NaN
        self.count_storage_matrix = np.zeros(
            (len(class_names), len(self.range_buckets)), dtype=np.int64
        )

    def accumulate_value(
        self,
        class_name: str,
        range_bucket: Tuple[float, float],
        average_epe: float,
        average_range: float,
        count: int,
    ):
        if count == 0 or np.isnan(average_epe) or np.isnan(average_range):
            print("Warning in accumulate_value: count is 0 or average_epe/average_range is NaN, skip this entry.")
            return
        # assert count > 0, f"count must be greater than 0, got {count}"
        # assert np.isfinite(average_epe), f"average_epe must be finite, got {average_epe}"
        # assert np.isfinite(average_range), f"average_range must be finite, got {average_range}"

        class_idx = self.class_names.index(class_name)
        range_bucket_idx = self.range_buckets.index(range_bucket)

        prior_epe = self.epe_storage_matrix[class_idx, range_bucket_idx]
        prior_speed = self.range_storage_matrix[class_idx, range_bucket_idx]
        prior_count = self.count_storage_matrix[class_idx, range_bucket_idx]

        if np.isnan(prior_epe):
            self.epe_storage_matrix[class_idx, range_bucket_idx] = average_epe
            self.range_storage_matrix[class_idx, range_bucket_idx] = average_range
            self.count_storage_matrix[class_idx, range_bucket_idx] = count
            return

        # Accumulate the average EPE and speed, weighted by the number of samples using np.mean
        self.epe_storage_matrix[class_idx, range_bucket_idx] = np.average(
            [prior_epe, average_epe], weights=[prior_count, count]
        )
        self.range_storage_matrix[class_idx, range_bucket_idx] = np.average(
            [prior_speed, average_range], weights=[prior_count, count]
        )
        self.count_storage_matrix[class_idx, range_bucket_idx] += count

    def get_class_entries(self, class_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_idx = self.class_names.index(class_name)

        epe = self.epe_storage_matrix[class_idx, :]
        range = self.range_storage_matrix[class_idx, :]
        count = self.count_storage_matrix[class_idx, :]
        return epe, range, count
    
    def get_normalized_error_matrix(self):
        pass

    def get_overall_class_errors(self, normalized: bool = True):
        pass

    def get_mean_average_values(self, normalized: bool = True):
        pass

class BucketedSpeedMatrix(BucketResultMatrix):
    def __init__(self, class_names: List[str], speed_buckets: List[Tuple[float, float]]):
        super().__init__(class_names, speed_buckets)

    def get_normalized_error_matrix(self):
        error_matrix = self.epe_storage_matrix.copy()
        # For the 1: columns, normalize EPE entries by speed (0 is static so we skip it)
        error_matrix[:, 1:] = error_matrix[:, 1:] / self.range_storage_matrix[:, 1:]
        return error_matrix

    def get_overall_class_errors(self, normalized: bool = True):
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

        self.epe_ssf = {} # will be like {"0-35": {"Static": [], "Dynamic": []}, "35-50": {"Static": [], "Dynamic": []}, ...}

        self.norm_flag = False
        self.ssf_eval = False

        # bucket_max_speed, num_buckets, distance_thresholds set is from: eval/bucketed_epe.py#L226
        speed_splits = np.concatenate([np.linspace(0, 2.0, 51), [np.inf]])
        self.bucketedMatrix = BucketedSpeedMatrix(
            class_names=['BACKGROUND', 'CAR', 'OTHER_VEHICLES', 'PEDESTRIAN', 'WHEELED_VRU'],
            speed_buckets=list(zip(speed_splits, speed_splits[1:]))
        )

        distance_split = [0, 35, 50, 75, 100, np.inf]
        self.distanceMatrix = BucketResultMatrix(
            class_names = ['Static', 'Dynamic'],
            range_buckets = list(zip(distance_split, distance_split[1:]))
        )
        for min_, max_ in list(zip(distance_split, distance_split[1:])):
            str_name = f"{int(min_)}-{int(max_)}" if max_ != np.inf else f"{int(min_)}-inf"
            self.epe_ssf[str_name] = {"Static": [], "Dynamic": [], "#Static": 0, "#Dynamic": 0}

    def step(self, epe_dict, bucket_dict, ssf_dict=None):
        """
        This step function is used to store the results of **each frame**.
        """
        for key in epe_dict:
            self.epe_3way[key].append(epe_dict[key])

        for item_ in bucket_dict:
            self.bucketedMatrix.accumulate_value(
                item_.name,
                item_.thresholds_range,
                item_.avg_epe,
                item_.avg_range,
                item_.count,
            )
        
        if ssf_dict is not None:
            # print("ssf_dict is not None")
            for item_ in ssf_dict:
                self.distanceMatrix.accumulate_value(
                    item_.name,
                    item_.thresholds_range,
                    item_.avg_epe,
                    item_.avg_range,
                    item_.count,
                )
    def normalize(self):
        """
        This normalize mean average results between **frame and frame**.
        """
        # epe 3-way evaluation
        for key in self.epe_3way:
            self.epe_3way[key] = np.mean(self.epe_3way[key])
        self.epe_3way['Three-way'] = np.mean([self.epe_3way['EPE_FD'], self.epe_3way['EPE_BS'], self.epe_3way['EPE_FS']])

        # bucketed evaluation
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

        # ssf evaluation
        self.epe_ssf['Mean'] = {"Static": [], "Dynamic": [], "#Static": np.nan, "#Dynamic": np.nan}
        
        for motion in ["Static", "Dynamic"]:
            avg_epes, avg_diss, num_pts = self.distanceMatrix.get_class_entries(motion)
            # print(avg_epe, avg_dis)
            for avg_epe, avg_dis, num_pt in zip(avg_epes, avg_diss, num_pts):
                for dis_range_key in self.epe_ssf:
                    if dis_range_key != 'Mean':
                        min_, max_ = dis_range_key.split("-")
                        min_, max_ = int(min_), int(max_) if max_ != "inf" else np.inf        
                        if max_ > avg_dis >= min_:
                            self.epe_ssf[dis_range_key][motion] = avg_epe
                            self.epe_ssf[dis_range_key]["#"+motion] += num_pt
            
            self.epe_ssf['Mean'][motion] = np.nanmean(avg_epes)

    def print(self, ssf_metrics: bool = False):
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
        print("Version 2 Metric on Normalized Category-based:")
        print(tabulate(printed_data, headers=["Class", "Static", "Dynamic"], tablefmt='orgtbl'), "\n")

        if ssf_metrics:
            printed_data = []
            for key in self.epe_ssf:
                printed_data.append([key, np.around(self.epe_ssf[key]['Static'],4), np.around(self.epe_ssf[key]['Dynamic'],4), self.epe_ssf[key]["#Static"], self.epe_ssf[key]["#Dynamic"]])
            print("Version 3 Metric on EPE Distance-based:")
            print(tabulate(printed_data, headers=["Distance", "Static", "Dynamic", "#Static", "#Dynamic"], tablefmt='orgtbl'), "\n")
