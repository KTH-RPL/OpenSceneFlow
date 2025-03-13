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
from src.utils.mics import import_func, weights_init, zip_res
from src.utils.av2_eval import write_output_file
from src.models.basic import cal_pose0to1
from src.utils.eval_metric import OfficialMetrics, evaluate_leaderboard, evaluate_leaderboard_v2, evaluate_ssf

# debugging tools
# import faulthandler
# faulthandler.enable()

torch.set_float32_matmul_precision('medium')
class ModelWrapper(LightningModule):
    def __init__(self, cfg, eval=False):
        super().__init__()

        # set grid size
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
        
        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)
        
        self.loss_fn = import_func("src.lossfuncs."+cfg.loss_fn) if 'loss_fn' in cfg else None
        self.add_seloss = cfg.add_seloss if 'add_seloss' in cfg else None
        self.cfg_loss_name = cfg.loss_fn if 'loss_fn' in cfg else None
        
        self.batch_size = int(cfg.batch_size) if 'batch_size' in cfg else 1
        self.lr = cfg.lr if 'lr' in cfg else None
        self.epochs = cfg.epochs if 'epochs' in cfg else None
        
        self.metrics = OfficialMetrics()

        self.load_checkpoint_path = cfg.checkpoint if 'checkpoint' in cfg else None


        self.leaderboard_version = cfg.leaderboard_version if 'leaderboard_version' in cfg else 1
        # NOTE(Qingwen): since we have seflow version which is unsupervised, we need to set the flag to false.
        self.supervised_flag = cfg.supervised_flag if 'supervised_flag' in cfg else True
        self.save_res = False
        if 'av2_mode' in cfg:
            self.av2_mode = cfg.av2_mode
            self.save_res = cfg.save_res if 'save_res' in cfg else False
            
            if self.save_res or self.av2_mode == 'test':
                self.save_res_path = Path(cfg.dataset_path).parent / "results" / cfg.output
                os.makedirs(self.save_res_path, exist_ok=True)
                print(f"We are in {cfg.av2_mode}, results will be saved in: {self.save_res_path} with version: {self.leaderboard_version} format for online leaderboard.")
        else:
            self.av2_mode = None
            if 'pretrained_weights' in cfg:
                if cfg.pretrained_weights is not None:
                    self.model.load_from_checkpoint(cfg.pretrained_weights)

        self.dataset_path = cfg.dataset_path if 'dataset_path' in cfg else None
        self.vis_name = cfg.res_name if 'res_name' in cfg else 'default'
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        self.model.timer[4].start("One Scan in model")
        res_dict = self.model(batch)
        self.model.timer[4].stop()

        self.model.timer[5].start("Loss")
        # compute loss
        total_loss = 0.0

        if self.cfg_loss_name in ['seflowLoss']:
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
                        'gt_classes': None if 'flow_category_indices' not in batch else batch['flow_category_indices'][batch_id][pc0_valid_from_pc2res]}
            
            if 'pc0_dynamic' in batch:
                dict2loss['pc0_labels'] = batch['pc0_dynamic'][batch_id][pc0_valid_from_pc2res]
                dict2loss['pc1_labels'] = batch['pc1_dynamic'][batch_id][pc1_valid_from_pc2res]

            # different methods may don't have this in the res_dict
            if 'pc0_points_lst' in res_dict and 'pc1_points_lst' in res_dict:
                dict2loss['pc0'] = pc0_points_lst[batch_id]
                dict2loss['pc1'] = pc1_points_lst[batch_id]

            res_loss = self.loss_fn(dict2loss)
            for i, loss_name in enumerate(loss_items):
                total_loss += weights[i] * res_loss[loss_name]
            for key in res_loss:
                loss_logger[key] += res_loss[key]

        self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        if self.add_seloss is not None and self.cfg_loss_name in ['seflowLoss']:
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
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self):
        self.time_start_train_epoch = time.time()

    def on_train_epoch_end(self):
        self.log("pre_epoch_cost (mins)", (time.time()-self.time_start_train_epoch)/60.0, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)

        if self.av2_mode == 'test':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
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
        
        if self.av2_mode == 'val':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            print(f"More details parameters and training status are in checkpoints")        

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
                pickle.dump((self.metrics.epe_3way, self.metrics.bucketed, self.metrics.epe_ssf, self.metrics.mean_num_occupied_voxels), f)
            print(f"We already write the {self.vis_name} into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
            print(f"python tools/visualization.py --res_name '{self.vis_name}' --data_dir {self.dataset_path}")
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

        if 'num_occupied_voxels' in res_dict:
            num_occupied_voxels = res_dict['num_occupied_voxels']
        else:
            num_occupied_voxels = None

        if self.av2_mode == 'val': # since only val we have ground truth flow to eval
            gt_flow = batch["flow"]
            v1_dict = evaluate_leaderboard(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                       gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], \
                                       batch['flow_category_indices'][eval_mask])
            v2_dict = evaluate_leaderboard_v2(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                    gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], batch['flow_category_indices'][eval_mask])
            ssf_dict = evaluate_ssf(final_flow, pose_flow, pc0, \
                                    gt_flow, batch['flow_is_valid'], batch['flow_category_indices'])
            self.metrics.step(v1_dict, v2_dict, ssf_dict, num_occupied_voxels)
        
        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        if self.save_res or self.av2_mode == 'test': # test must save data to submit in the online leaderboard.    
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
        if 'pcb0' in batch:
            batch['pcb0'] = batch['pcb0'][~batch['gmb0']].unsqueeze(0)
        self.model.timer[12].start("One Scan")
        res_dict = self.model(batch)
        self.model.timer[12].stop()

        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if res_dict[key]!=None and len(res_dict[key])>0}
        return batch, res_dict
    
    def validation_step(self, batch, batch_idx):
        if self.av2_mode == 'val' or self.av2_mode == 'test':
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
            if self.vis_name in f[key]:
                del f[key][self.vis_name]
            f[key].create_dataset(self.vis_name, data=final_flow.cpu().detach().numpy().astype(np.float32))

    def on_test_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)
        print(f"\n\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
        print(f"We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
        print(f"python tools/visualization.py --res_name '{self.vis_name}' --data_dir {self.dataset_path}")
        print(f"Enjoy! ^v^ ------ \n")
