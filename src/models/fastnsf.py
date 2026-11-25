
"""
This file is from: https://github.com/Lilac-Lee/FastNSF
with slightly modification to have unified format with all benchmark.

# Created: 2024-07-27 11:40
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * SeFlow (https://github.com/KTH-RPL/SeFlow) 
# * HiMo (https://kin-zhang.github.io/HiMo)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: FastNSF to our codebase implementation.
# one more package need install: pip install FastGeodis==1.0.4 --no-build-isolation
"""

import dztimer, torch
import torch.nn as nn

import FastGeodis # extra package, the distance transform (DT) needs it

from .basic.opt_module import Neural_Prior, EarlyStopping, DT
from .basic import cal_pose0to1, in_range

class FastNSF(nn.Module):
    def __init__(self, filter_size=128, act_fn='relu', layer_size=8, grid_factor=10.,\
                 itr_num=5000, lr=8e-3, min_delta=0.00005, early_patience=30,
                 verbose=False, point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3], init_weight = True):
        super().__init__()
        
        self.filter_size = filter_size
        self.act_fn = act_fn
        self.layer_size = layer_size

        self.grid_factor = grid_factor # grid cell size=1/grid_factor.

        self.iteration_num = itr_num
        self.min_delta = min_delta
        self.lr = lr
        self.early_patience = early_patience
        self.verbose = verbose
        self.point_cloud_range = point_cloud_range
        self.timer = dztimer.Timing()
        self.timer.start("NSFP Model Inference")
        self.init_weight = init_weight
        print(f"\n---LOG [model]: FastNSF setup itr_num: {itr_num}, lr: {lr}, early_patience: {early_patience}.")
            
    def optimize(self, dict2loss):
        device = dict2loss['pc0'].device

        # NOTE(Qingwen): don't know why, but it must be initialized every optimization time.
        self.timer[5].start("Network Initialization")
        net = Neural_Prior(filter_size=self.filter_size, act_fn=self.act_fn, layer_size=self.layer_size)
        net = net.to(device)
        if self.init_weight:
            net.init_weights()
        net.train()
        self.timer[5].stop()

        pc0 = dict2loss['pc0']
        pc1 = dict2loss['pc1']

        pc1_min = torch.min(pc0.squeeze(0), 0)[0]
        pc2_min = torch.min(pc1.squeeze(0), 0)[0]
        pc1_max = torch.max(pc0.squeeze(0), 0)[0]
        pc2_max = torch.max(pc1.squeeze(0), 0)[0]
        
        xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min<pc2_min, pc1_min, pc2_min) * self.grid_factor-1) / self.grid_factor
        xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max>pc2_max, pc1_max, pc2_max)* self.grid_factor+1) / self.grid_factor
        # print('xmin: {}, xmax: {}, ymin: {}, ymax: {}, zmin: {}, zmax: {}'.format(xmin_int, xmax_int, ymin_int, ymax_int, zmin_int, zmax_int))
        
        # NOTE: build DT map
        dt = DT(pc1.clone().squeeze(0).to(device), (xmin_int, ymin_int, zmin_int), (xmax_int, ymax_int, zmax_int), self.grid_factor, device)
        params = net.parameters()
        best_forward = {'loss': torch.inf}

        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=0)
        early_stopping = EarlyStopping(patience=self.early_patience, min_delta=self.min_delta)

        for itr_ in range(self.iteration_num):
            optimizer.zero_grad()
            self.timer[1].start("Network Time")

            self.timer[1][0].start("Forward")
            forward_flow = net(pc0)
            self.timer[1][0].stop()
            pc0_to_pc1 = pc0 + forward_flow
            self.timer[1].stop()

            self.timer[2].start("loss")
            loss = dt.torch_bilinear_distance(pc0_to_pc1.squeeze(0)).mean()
            self.timer[2].stop()

            if loss <= best_forward['loss']:
                best_forward['loss'] = loss.item()
                best_forward['flow'] = pc0_to_pc1 - pc0

            if early_stopping.step(loss) and 'flow' in best_forward: # at least one step
                break

            self.timer[3].start("Loss Backward")
            loss.backward()
            self.timer[3].stop()

            self.timer[4].start("Optimizer Step")
            optimizer.step()
            self.timer[4].stop()

        if self.verbose:
            self.timer.print(random_colors=True, bold=True)

        return best_forward
    
    def forward(self, batch):
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        batch_final_flow = []
        for batch_id in range(batch_sizes):
            self.timer[0].start("Data Processing")
            pc0 = batch["pc0"][batch_id]
            pc1 = batch["pc1"][batch_id]
            rm0, rm1 = in_range(pc0, self.point_cloud_range), in_range(pc1, self.point_cloud_range)
            selected_pc0, selected_pc1 = pc0[rm0], pc1[rm1]
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

            # since pl in val and test mode will disable_grad.
            with torch.inference_mode(False):
                with torch.enable_grad():
                    dict2loss = {
                        'pc0': transform_pc0.clone().detach(), #.requires_grad_(True),
                        'pc1': selected_pc1.clone().detach() #.requires_grad_(True)
                    }
                    model_res = self.optimize(dict2loss)
            
            final_flow = torch.zeros_like(pc0)
            final_flow[rm0] = model_res["flow"].clone().detach().requires_grad_(False)
            batch_final_flow.append(final_flow)

        res_dict = {"flow": batch_final_flow,
                    "pose_flow": pose_flows
                    }
        
        return res_dict