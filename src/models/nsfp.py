
"""
This file is from: https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior
with our modification to have unified format with our codebase running.

# Created: 2024-07-27 11:33
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * SeFlow (https://github.com/KTH-RPL/SeFlow) 
# * HiMo (https://kin-zhang.github.io/HiMo)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: NSFP to our codebase implementation.

"""

import dztimer, torch, copy
import torch.nn as nn

from .basic import cal_pose0to1
from .basic.opt_module import Neural_Prior, EarlyStopping
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()
                
class NSFP(nn.Module):
    def __init__(self, filter_size=128, act_fn='relu', layer_size=8, \
                 itr_num=5000, lr=8e-3, min_delta=0.00005, early_patience=30,
                 verbose=False, point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3]):
        super().__init__()
        
        self.filter_size = filter_size
        self.act_fn = act_fn
        self.layer_size = layer_size

        self.iteration_num = itr_num
        self.min_delta = min_delta
        self.lr = lr
        self.early_patience = early_patience
        self.verbose = verbose
        self.point_cloud_range = point_cloud_range
        self.timer = dztimer.Timing()
        self.timer.start("NSFP Model Inference")
        print(f"\n--- LOG [model]: NSFP set itr_num: {itr_num}, lr: {lr}, early_patience: {early_patience}.")
    
    def cal_loss(self, dict2loss, net_inv, pc0_to_pc1):

        self.timer[2].start("Loss")


        self.timer[1][1].start("Inverse")
        inverse_flow = net_inv(pc0_to_pc1)
        self.timer[1][1].stop()
        est_pc1_to_pc0 = pc0_to_pc1 - inverse_flow

            
        self.timer[2][0].start("Forward Loss")
        # forward_loss, _ = my_chamfer_fn(pc0_to_pc1.unsqueeze(0), dict2loss['pc1'].unsqueeze(0))
        forward_loss = MyCUDAChamferDis.truncated_dis(pc0_to_pc1, dict2loss['pc1'])
        self.timer[2][0].stop()

        self.timer[2][1].start("Inverse Loss")
        # inverse_loss, _ = my_chamfer_fn(est_pc1_to_pc0.unsqueeze(0), dict2loss['pc0'].unsqueeze(0))
        inverse_loss = MyCUDAChamferDis.truncated_dis(est_pc1_to_pc0, dict2loss['pc0'])
        self.timer[2][1].stop()
        loss = forward_loss + inverse_loss

        self.timer[2].stop()

        return loss
            
    def optimize(self, dict2loss):
        device = dict2loss['pc0'].device

        # NOTE(Qingwen): don't know why, but it must be initialized every optimization time.
        self.timer[5].start("Network Initialization")
        net = Neural_Prior(filter_size=self.filter_size, act_fn=self.act_fn, layer_size=self.layer_size)
        net = net.to(device)
        net.train()
        net_inv = copy.deepcopy(net)
        self.timer[5].stop()

        pc0 = dict2loss['pc0']
        params = [{
            'params': net.parameters(),
            'lr': self.lr,
            'weight_decay': 0
        }, {
            'params': net_inv.parameters(),
            'lr': self.lr,
            'weight_decay': 0
        }]
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

            dict2loss['est_flow'] = forward_flow
            loss = self.cal_loss(dict2loss, net_inv, pc0_to_pc1)
            
            if loss <= best_forward['loss']:
                best_forward['loss'] = loss.item()
                best_forward['flow'] = forward_flow

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
    
    def range_limit_(self, pc):
        """
        Limit the point cloud to the given range.
        """
        mask = (pc[:, 0] >= self.point_cloud_range[0]) & (pc[:, 0] <= self.point_cloud_range[3]) & \
               (pc[:, 1] >= self.point_cloud_range[1]) & (pc[:, 1] <= self.point_cloud_range[4]) & \
               (pc[:, 2] >= self.point_cloud_range[2]) & (pc[:, 2] <= self.point_cloud_range[5])
        return pc[mask], mask
    
    def forward(self, batch):
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        batch_final_flow = []
        for batch_id in range(batch_sizes):
            self.timer[0].start("Data Processing")
            pc0 = batch["pc0"][batch_id]
            pc1 = batch["pc1"][batch_id]
            selected_pc0, rm0 = self.range_limit_(pc0)
            selected_pc1, rm1 = self.range_limit_(pc1)
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
                        'pc0': transform_pc0.clone().detach().requires_grad_(True),
                        'pc1': selected_pc1.clone().detach().requires_grad_(True)
                    }
                    model_res = self.optimize(dict2loss)
            
            final_flow = torch.zeros_like(pc0)
            final_flow[rm0] = model_res["flow"].clone().detach().requires_grad_(False)
            batch_final_flow.append(final_flow)

        res_dict = {"flow": batch_final_flow,
                    "pose_flow": pose_flows
                    }
        
        return res_dict