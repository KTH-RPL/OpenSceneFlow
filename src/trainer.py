"""

# Created: 2023-11-05 10:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Model Wrapper for Pytorch Lightning

"""

import torch
import torch.optim as optim

from lightning import LightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

import os, sys, time
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils import import_func
from src.lossfuncs import SSL_LOSSES_FN
from src.utils.mics import weights_init, enable_timer_sync
from src.models.basic import cal_pose0to1, WarmupCosLR
from src.utils.eval_metric import OfficialMetrics, evaluate_leaderboard, evaluate_leaderboard_v2, evaluate_ssf

# debugging tools
# import faulthandler
# faulthandler.enable()

torch.set_float32_matmul_precision('medium')
class ModelWrapper(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        default_self_values = {
            "batch_size": 1,
            "lr": 2e-4,
            "epochs": 3,
            "loss_fn": 'deflowLoss',
            "add_seloss": None,

            # lr scheduler, only active when warmup_epochs > 0
            "optimizer": None,
            "cluster_loss_args": {},
            "loss_frames": None,
            "timer_sync": False,
        }
        for key, default in default_self_values.items():
            setattr(self, key, cfg.get(key, default))

        if ('voxel_size' in cfg.model.target) and ('point_cloud_range' in cfg.model.target) and 'point_cloud_range' in cfg:
            OmegaConf.set_struct(cfg.model.target, True)
            with open_dict(cfg.model.target):
                cfg.model.target['grid_feature_size'] = \
                    [abs(int((cfg.point_cloud_range[0] - cfg.point_cloud_range[3]) / cfg.voxel_size[0])),
                    abs(int((cfg.point_cloud_range[1] - cfg.point_cloud_range[4]) / cfg.voxel_size[1])),
                    abs(int((cfg.point_cloud_range[2] - cfg.point_cloud_range[5]) / cfg.voxel_size[2]))]
        else:
            with open_dict(cfg.model.target):
                cfg.model.target['grid_feature_size'] = \
                    [abs(int((cfg.model.target.point_cloud_range[0] - cfg.model.target.point_cloud_range[3]) / cfg.model.target.voxel_size[0])),
                    abs(int((cfg.model.target.point_cloud_range[1] - cfg.model.target.point_cloud_range[4]) / cfg.model.target.voxel_size[1])),
                    abs(int((cfg.model.target.point_cloud_range[2] - cfg.model.target.point_cloud_range[5]) / cfg.model.target.voxel_size[2]))]
        
        # ---> model
        self.point_cloud_range = cfg.model.target.point_cloud_range
        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)
        if 'pretrained_weights' in cfg and cfg.pretrained_weights is not None:
            missing_keys, unexpected_keys = self.model.load_from_checkpoint(cfg.pretrained_weights)
        # print(f"Model: {self.model.__class__.__name__}, Number of Frames: {self.num_frames}")
        # NOTE(Qingwen): dztimer only reads the CPU clock, timer_sync=True syncs cuda at every start/stop to get the real runtime of each part.
        # It's off by default since the sync slows down training.
        if self.timer_sync:
            enable_timer_sync(self.model)

        # ---> loss fn
        self.loss_fn = import_func("src.lossfuncs."+cfg.loss_fn) if 'loss_fn' in cfg else None
        self.cfg_loss_name = cfg.get("loss_fn", None)
        
        # ---> evaluation metric
        self.metrics = OfficialMetrics()

        self.save_hyperparameters()
        
    def ssl_loss_calculator(self, batch, res_dict, if_log=True):
        """Build dict2loss for ALL self-supervised losses (seflow, seflowpp, teflow*).

        Each frame is represented only as a List[Tensor] and a List[labels].
        No flat tensors, no offsets, no sizes — chamfer calls use list APIs only.
        """
        total_loss, bz_ = 0.0, len(batch["pose0"])

        pc0_list = [res_dict['pc0_points_lst'][i] for i in range(bz_)]

        dict2loss = {
            'pc0_list':         pc0_list,
            'est_flow_list':    [res_dict['flow'][i] for i in range(bz_)],
            'pc0_labels_list':  [batch['pc0_dynamic'][i][res_dict['pc0_valid_point_idxes'][i]] for i in range(bz_)],
            'batch_size':       bz_,
        }

        frame_keys = [key.replace('_points_lst', '') for key in res_dict.keys()
                      if key.startswith('pc') and key.endswith('_points_lst')]
        frame_keys.remove('pc0')
        if self.loss_frames is not None:
            # the loss supervises with exactly these frames, e.g. {past: 2, future: 2} -> pch1, pch2, pc1, pc2
            frame_keys = [f'pch{i}' for i in range(1, self.loss_frames['past'] + 1)] \
                       + [f'pc{i}' for i in range(1, self.loss_frames['future'] + 1)]

        for frame_id in frame_keys:
            if f'{frame_id}_points_lst' in res_dict: # the model took this frame as input
                points_list = [res_dict[f'{frame_id}_points_lst'][i] for i in range(bz_)]
                labels_list = [batch[f'{frame_id}_dynamic'][i][res_dict[f'{frame_id}_valid_point_idxes'][i]] for i in range(bz_)]
            else:
                points_list, labels_list = map(list, zip(*[self.loss_only_frame(batch, frame_id, i) for i in range(bz_)]))
            dict2loss[f'{frame_id}_list']        = points_list
            dict2loss[f'{frame_id}_labels_list'] = labels_list
                    
        loss_items, weights = zip(*[(key, weight) for key, weight in self.add_seloss.items()])
        dict2loss['loss_weights_dict'] = self.add_seloss
        
        dict2loss['cluster_loss_args'] = self.cluster_loss_args

        res_loss = self.loss_fn(dict2loss)

        for i, loss_name in enumerate(loss_items):
            if not torch.isnan(res_loss[loss_name]):
                total_loss += weights[i] * res_loss[loss_name]
        
        if if_log:
            self.log("trainer/loss", total_loss, sync_dist=True, batch_size=bz_, prog_bar=True)
        for key in res_loss:
            self.log(f"trainer/{key}", res_loss[key], sync_dist=True, batch_size=bz_)
            
        return total_loss

    def loss_only_frame(self, batch, frame_id, batch_id):
        """Points and auto-labels of a frame that only the loss uses (see loss_frames), e.g. 'pc2' or 'pch2'.

        Same as what the model does to its input frames: move the points to pc1 coordinate and keep the ones inside point_cloud_range.
        """
        pose_to_pc1 = cal_pose0to1(batch[frame_id.replace('pc', 'pose')][batch_id], batch['pose1'][batch_id])
        points = batch[frame_id][batch_id] @ pose_to_pc1[:3, :3].T + pose_to_pc1[:3, 3]
        # NOTE: the padding points in a batch are nan, so they are also removed here.
        pc_range = points.new_tensor(list(self.point_cloud_range))
        in_range = ((points >= pc_range[:3]) & (points < pc_range[3:])).all(dim=1)
        return points[in_range], batch[f'{frame_id}_dynamic'][batch_id][in_range]

    def loss_calculator(self, batch, res_dict, if_log=True):
        """ Calculate the loss based on the batch (gt/ssl-label) and res_dict (estimate flow)."""
        def get_batch_data(batch, key, batch_id, batch_sizes, pc0_valid_from_pc2res, pose_flow_=None):
            """NOTE(Qingwen): for gt need double check whether it exists in the batch and batch size is correct"""
            if key not in batch or batch[key].shape[0] != batch_sizes:
                return None
            data = batch[key][batch_id][pc0_valid_from_pc2res]
            if key == 'flow' and pose_flow_ is not None:
                data = data - pose_flow_
            return data
        def get_frame_keys(data_dict, suffix):
            return [key for key in data_dict.keys() if key.endswith(suffix)]
        def extract_frame_id(key, suffix):
            """Extract frame identifier from key (e.g., 'pc0_points_lst' -> 'pc0')"""
            return key.replace(suffix, '')
        
        # Supervised-only path (deflowLoss, etc.)
        # SSL losses are handled by ssl_loss_calculator.
        total_loss, loss_logger = 0.0, {}
        loss_items, weights = ['loss'], [1.0]
        for key in loss_items:
            loss_logger[key] = 0.0

        batch_sizes, pose_flows, est_flow = len(batch["pose0"]), res_dict['pose_flow'], res_dict['flow']
        for batch_id in range(batch_sizes):
            # Get pc0 valid indices (main reference frame)
            pc0_valid_from_pc2res = res_dict['pc0_valid_point_idxes'][batch_id]
            pose_flow_ = pose_flows[batch_id][pc0_valid_from_pc2res]

            dict2loss = {'est_flow': est_flow[batch_id], 
                        'gt_flow': get_batch_data(batch, 'flow', batch_id, batch_sizes, pc0_valid_from_pc2res, pose_flow_),
                        'gt_classes': get_batch_data(batch, 'flow_category_indices', batch_id, batch_sizes, pc0_valid_from_pc2res),
                        'gt_instance': get_batch_data(batch, 'flow_instance_id', batch_id, batch_sizes, pc0_valid_from_pc2res)}
            
            # Add all available point cloud frames
            for points_key in get_frame_keys(res_dict, '_points_lst'):
                frame_id = extract_frame_id(points_key, '_points_lst')
                if points_key in res_dict:
                    dict2loss[frame_id] = res_dict[points_key][batch_id]

            res_loss = self.loss_fn(dict2loss)
 
            for i, loss_name in enumerate(loss_items):
                # if torch.isnan(res_loss[loss_name]):
                #     print(f"==> Loss: {loss_name} is nan, skip this batch.")
                #     continue
                total_loss += weights[i] * res_loss[loss_name]
            for key in res_loss:
                loss_logger[key] += res_loss[key]
        if if_log:
            self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        return total_loss
    
    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        self.model.timer[5].start("Training Step")
        self.model.timer[5][0].start("Forward")
        res_dict = self.model(batch)
        self.model.timer[5][0].stop()
        self.model.timer[5][1].start("Compute Loss")

        if self.cfg_loss_name in SSL_LOSSES_FN:
            total_loss = self.ssl_loss_calculator(batch, res_dict)
        else:
            total_loss = self.loss_calculator(batch, res_dict)
        self.model.timer[5][1].stop()
        self.model.timer[5].stop()
        
        # NOTE (Qingwen): if you want to view the detail breakdown of time cost
        # self.model.timer.print(random_colors=False, bold=False)
        return total_loss

    def configure_optimizers(self):
        optimizers_ = {}
        # default Adam
        if self.optimizer.name == "AdamW":
            optimizers_['optimizer'] = optim.AdamW(self.model.parameters(), lr=self.optimizer.lr, weight_decay=self.optimizer.get("weight_decay", 1e-4))
        else: # if self.optimizer.name == "Adam":
            optimizers_['optimizer'] = optim.Adam(self.model.parameters(), lr=self.optimizer.lr)

        if "scheduler" in self.optimizer:
            if self.optimizer.scheduler.name == "WarmupCosLR":
                optimizers_['lr_scheduler'] = WarmupCosLR(optimizers_['optimizer'], self.optimizer.scheduler.get("min_lr", self.optimizer.lr*0.1), \
                                        self.optimizer.lr, self.optimizer.scheduler.get("warmup_epochs", 1), self.epochs)
            elif self.optimizer.scheduler.name == "StepLR":
                optimizers_['lr_scheduler'] = optim.lr_scheduler.StepLR(optimizers_['optimizer'], step_size=self.optimizer.scheduler.get("step_size", self.trainer.max_epochs//3), \
                                        gamma=self.optimizer.scheduler.get("gamma", 0.1))

        return optimizers_

    def on_train_epoch_start(self):
        self.time_start_train_epoch = time.time()

    def on_train_epoch_end(self):
        self.log("pre_epoch_cost (mins)", (time.time()-self.time_start_train_epoch)/60.0, on_step=False, on_epoch=True, sync_dist=True)
        # # NOTE (Qingwen): if you want to view the detail breakdown of time cost, set timer_sync=True
        if self.timer_sync and self.trainer.is_global_zero:
            self.model.timer.print(random_colors=False, bold=False)
    
    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)

        self.metrics.normalize()

        # wandb log things:
        for key in self.metrics.bucketed:
            for type_ in 'Static', 'Dynamic':
                self.log(f"val/{type_}/{key}", self.metrics.bucketed[key][type_], sync_dist=True)
        for key in self.metrics.epe_3way:
            self.log(f"val/{key}", self.metrics.epe_3way[key], sync_dist=True)
        
        self.metrics.print()

        self.metrics = OfficialMetrics()

    def validation_step(self, batch, batch_idx):
        try:
            res_dict = self.model(batch)
            if batch['flow'][0].shape[0] == 0: # no ground truth flow, nothing to evaluate
                return
            pose_flows = res_dict['pose_flow']
            for batch_id, gt_flow in enumerate(batch["flow"]):
                valid_from_pc2res = res_dict['pc0_valid_point_idxes'][batch_id]
                pose_flow = pose_flows[batch_id][valid_from_pc2res]

                final_flow_ = pose_flow.clone() + res_dict['flow'][batch_id]
                v1_dict= evaluate_leaderboard(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                v2_dict = evaluate_leaderboard_v2(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                ssf_dict = evaluate_ssf(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                self.metrics.step(v1_dict, v2_dict, ssf_dict)
        except Exception as e:
            print(f"==> Exception occur during training/validation step: {e}. Skip this batch.")
            print(f"Batch info: scene_id: {batch['scene_id']}, timestamp: {batch['timestamp']}, pc0 size: {batch['pc0']}")
