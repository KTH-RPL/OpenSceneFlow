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

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

from lightning import LightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf,open_dict

import os, sys, time, h5py, pickle
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils import import_func
from src.utils.mics import weights_init, zip_res
from src.utils.av2_eval import write_output_file
from src.models.basic import cal_pose0to1, WarmupCosLR
from src.utils.eval_metric import OfficialMetrics, evaluate_leaderboard, evaluate_leaderboard_v2, evaluate_ssf

# debugging tools
# import faulthandler
# faulthandler.enable()

torch.set_float32_matmul_precision('medium')
class ModelWrapper(LightningModule):
    def __init__(self, cfg, eval=False):
        super().__init__()

        default_self_values = {
            "batch_size": 1,
            "lr": 2e-4,
            "epochs": 3,
            "loss_fn": 'deflowLoss',
            "add_seloss": None,
            "checkpoint": None,
            "leaderboard_version": 2,
            "supervised_flag": True,
            "save_res": False,
            "res_name": "default",
            "num_frames": 2,
            "optimizer": None,
            "dataset_path": None,
            "data_mode": None,
        }
        for key, default in default_self_values.items():
            setattr(self, key, cfg.get(key, default))

        if ('voxel_size' in cfg.model.target) and ('point_cloud_range' in cfg.model.target) and not eval and 'point_cloud_range' in cfg:
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

        # ---> loss fn
        self.loss_fn = import_func("src.lossfuncs."+cfg.loss_fn) if 'loss_fn' in cfg else None
        self.cfg_loss_name = cfg.get("loss_fn", None)
        
        # ---> evaluation metric
        self.metrics = OfficialMetrics()

        # ---> inference mode
        if self.save_res and self.data_mode in ['val', 'valid', 'test']:
            self.save_res_path = Path(cfg.dataset_path).parent / "results" / cfg.output
            os.makedirs(self.save_res_path, exist_ok=True)
            print(f"We are in {cfg.data_mode}, results will be saved in: {self.save_res_path} with version: {self.leaderboard_version} format for online leaderboard.")

        # self.test_total_num = 0
        if self.data_mode in ['val', 'valid', 'test']:
            print(cfg)
        self.save_hyperparameters()

    # FIXME(Qingwen 2025-08-20): update the loss_calculation fn alone to make all things pretty here....
    def training_step(self, batch, batch_idx):
        self.model.timer[4].start("One Scan in model")
        res_dict = self.model(batch)
        self.model.timer[4].stop()

        self.model.timer[5].start("Loss")
        # compute loss
        total_loss = 0.0

        if self.cfg_loss_name in ['seflowLoss', 'seflowppLoss']:
            loss_items, weights = zip(*[(key, weight) for key, weight in self.add_seloss.items()])
            loss_logger = {'chamfer_dis': 0.0, 'dynamic_chamfer_dis': 0.0, 'static_flow_loss': 0.0, 'cluster_based_pc0pc1': 0.0}
        else:
            loss_items, weights = ['loss'], [1.0]
            loss_logger = {'loss': 0.0}

        pc0_valid_idx = res_dict['pc0_valid_point_idxes'] # since padding
        pc1_valid_idx = res_dict['pc1_valid_point_idxes'] # since padding
        if 'pc0_points_lst' in res_dict and 'pc1_points_lst' in res_dict:
            pc0_points_lst = res_dict['pc0_points_lst']
            pc1_points_lst = res_dict['pc1_points_lst']
        
        batch_sizes = len(batch["pose0"])
        pose_flows = res_dict['pose_flow']
        est_flow = res_dict['flow']
        
        for batch_id in range(batch_sizes):
            pc0_valid_from_pc2res = pc0_valid_idx[batch_id]
            pc1_valid_from_pc2res = pc1_valid_idx[batch_id]
            pose_flow_ = pose_flows[batch_id][pc0_valid_from_pc2res]

            dict2loss = {'est_flow': est_flow[batch_id], 
                        'gt_flow': None if 'flow' not in batch else batch['flow'][batch_id][pc0_valid_from_pc2res] - pose_flow_, 
                        'gt_classes': None if 'flow_category_indices' not in batch else batch['flow_category_indices'][batch_id][pc0_valid_from_pc2res],
                        'gt_instance': None if 'flow_instance_id' not in batch else batch['flow_instance_id'][batch_id][pc0_valid_from_pc2res],}
            
            if 'pc0_dynamic' in batch:
                dict2loss['pc0_labels'] = batch['pc0_dynamic'][batch_id][pc0_valid_from_pc2res]
                dict2loss['pc1_labels'] = batch['pc1_dynamic'][batch_id][pc1_valid_from_pc2res]
            if 'pch1_dynamic' in batch and 'pch1_valid_point_idxes' in res_dict:
                dict2loss['pch1_labels'] = batch['pch1_dynamic'][batch_id][res_dict['pch1_valid_point_idxes'][batch_id]]

            # different methods may don't have this in the res_dict
            if 'pc0_points_lst' in res_dict and 'pc1_points_lst' in res_dict:
                dict2loss['pc0'] = pc0_points_lst[batch_id]
                dict2loss['pc1'] = pc1_points_lst[batch_id]
            if 'pch1_points_lst' in res_dict:
                dict2loss['pch1'] = res_dict['pch1_points_lst'][batch_id]

            res_loss = self.loss_fn(dict2loss)
            for i, loss_name in enumerate(loss_items):
                total_loss += weights[i] * res_loss[loss_name]
            for key in res_loss:
                loss_logger[key] += res_loss[key]

        self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        if self.add_seloss is not None and self.cfg_loss_name in ['seflowLoss', 'seflowppLoss']:
            for key in loss_logger:
                self.log(f"trainer/{key}", loss_logger[key]/batch_sizes, sync_dist=True, batch_size=self.batch_size)
        self.model.timer[5].stop()

        # NOTE (Qingwen): if you want to view the detail breakdown of time cost
        # self.model.timer.print(random_colors=False, bold=False)
        return total_loss

    def train_validation_step_(self, batch, res_dict):
        # means there are ground truth flow so we can evaluate the EPE-3 Way metric
        if batch['flow'][0].shape[0] > 0:
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
        else:
            pass

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
    
    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)

        if self.data_mode == 'test':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.checkpoint}")
            print(f"Test results saved in: {self.save_res_path}, Please run submit command and upload to online leaderboard for results.")
            if self.leaderboard_version == 1:
                print(f"\nevalai challenge 2010 phase 4018 submit --file {self.save_res_path}.zip --large --private\n")
            elif self.leaderboard_version == 2:
                print(f"\nevalai challenge 2210 phase 4396 submit --file {self.save_res_path}.zip --large --private\n")
            else:
                print(f"Please check the leaderboard version in the config file. We only support version 1 and 2.")
            output_file = zip_res(self.save_res_path, leaderboard_version=self.leaderboard_version, is_supervised = self.supervised_flag, output_file=self.save_res_path.as_posix() + ".zip")
            # wandb.log_artifact(output_file)
            return
        
        if self.data_mode == 'val':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.checkpoint}")
            print(f"More details parameters and training status are in the checkpoint file.")        

        self.metrics.normalize()

        # wandb log things:
        for key in self.metrics.bucketed:
            for type_ in 'Static', 'Dynamic':
                self.log(f"val/{type_}/{key}", self.metrics.bucketed[key][type_], sync_dist=True)
        for key in self.metrics.epe_3way:
            self.log(f"val/{key}", self.metrics.epe_3way[key], sync_dist=True)
        
        self.metrics.print()

        if self.save_res:
            # Save the dictionaries to a pickle file
            with open(str(self.save_res_path)+'.pkl', 'wb') as f:
                pickle.dump((self.metrics.epe_3way, self.metrics.bucketed, self.metrics.epe_ssf), f)
            print(f"We already write the {self.res_name} into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
            print(f"python tools/visualization.py --res_name '{self.res_name}' --data_dir {self.dataset_path}")
            print(f"Enjoy! ^v^ ------ \n")

        self.metrics = OfficialMetrics()
        
    def eval_only_step_(self, batch, res_dict):
        eval_mask = batch['eval_mask'].squeeze()
        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        final_flow = pose_flow.clone()
        if 'pc0_valid_point_idxes' in res_dict:
            valid_from_pc2res = res_dict['pc0_valid_point_idxes']

            # flow in the original pc0 coordinate
            pred_flow = pose_flow[~batch['gm0']].clone()
            pred_flow[valid_from_pc2res] = res_dict['flow'] + pose_flow[~batch['gm0']][valid_from_pc2res]
            final_flow[~batch['gm0']] = pred_flow
        else:
            final_flow[~batch['gm0']] = res_dict['flow'] + pose_flow[~batch['gm0']]

        if self.data_mode == 'val': # since only val we have ground truth flow to eval
            gt_flow = batch["flow"]
            v1_dict = evaluate_leaderboard(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                       gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], \
                                       batch['flow_category_indices'][eval_mask])
            v2_dict = evaluate_leaderboard_v2(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                    gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], batch['flow_category_indices'][eval_mask])
            ssf_dict = evaluate_ssf(final_flow, pose_flow, pc0, \
                                    gt_flow, batch['flow_is_valid'], batch['flow_category_indices'])
            self.metrics.step(v1_dict, v2_dict, ssf_dict)
        
        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        if self.save_res or self.data_mode == 'test': # test must save data to submit in the online leaderboard.    
            save_pred_flow = final_flow[eval_mask, :3].cpu().detach().numpy()
            rigid_flow = pose_flow[eval_mask, :3].cpu().detach().numpy()
            is_dynamic = np.linalg.norm(save_pred_flow - rigid_flow, axis=1, ord=2) >= 0.05
            sweep_uuid = (batch['scene_id'], batch['timestamp'])
            if self.leaderboard_version == 2:
                save_pred_flow = (final_flow - pose_flow).cpu().detach().numpy() # all points here... since 2rd version we need to save the relative flow.
            write_output_file(save_pred_flow, is_dynamic, sweep_uuid, self.save_res_path, leaderboard_version=self.leaderboard_version)

    def run_model_wo_ground_data(self, batch):
        # NOTE (Qingwen): only needed when val or test mode, since train we will go through collate_fn to remove.
        batch['origin_pc0'] = batch['pc0'].clone()
        batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
        batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
        
        for i in range(1, self.num_frames-1):
            batch[f'pch{i}'] = batch[f'pch{i}'][~batch[f'gmh{i}']].unsqueeze(0)

        self.model.timer[12].start("One Scan")
        res_dict = self.model(batch)
        self.model.timer[12].stop()

        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if res_dict[key]!=None and len(res_dict[key])>0}
        return batch, res_dict
    
    def validation_step(self, batch, batch_idx):
        if self.data_mode in ['val', 'test']:
            batch, res_dict = self.run_model_wo_ground_data(batch)
            self.model.timer[13].start("Eval")
            self.eval_only_step_(batch, res_dict)
            self.model.timer[13].stop()
        else:
            res_dict = self.model(batch)
            self.train_validation_step_(batch, res_dict)

    def test_step(self, batch, batch_idx):
        batch, res_dict = self.run_model_wo_ground_data(batch)
        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        final_flow = pose_flow.clone()
        if 'pc0_valid_point_idxes' in res_dict:
            valid_from_pc2res = res_dict['pc0_valid_point_idxes']

            # flow in the original pc0 coordinate
            pred_flow = pose_flow[~batch['gm0']].clone()
            pred_flow[valid_from_pc2res] = pose_flow[~batch['gm0']][valid_from_pc2res] + res_dict['flow']

            final_flow[~batch['gm0']] = pred_flow
        else:
            final_flow[~batch['gm0']] = res_dict['flow'] + pose_flow[~batch['gm0']]

        # write final_flow into the dataset.
        key = str(batch['timestamp'])
        scene_id = batch['scene_id']
        with h5py.File(os.path.join(self.dataset_path, f'{scene_id}.h5'), 'r+') as f:
            if self.res_name in f[key]:
                del f[key][self.res_name]
            f[key].create_dataset(self.res_name, data=final_flow.cpu().detach().numpy().astype(np.float32))

    def on_test_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)
        print(f"\n\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.checkpoint}")
        print(f"We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
        print(f"python tools/visualization.py --res_name '{self.res_name}' --data_dir {self.dataset_path}")
        print(f"Enjoy! ^v^ ------ \n")
