
"""
# Created: 2024-05-16 15:08
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Ajinkya Khoche (https://ajinkyakhoche.github.io/)
#
# This file is part of SSF (https://github.com/KTH-RPL/SSF).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import torch.nn as nn
import dztimer, torch

from .basic.sparse_unet import SimpleSparseUNet
from .basic.encoder import DynamicVoxelizer
from .basic.ssf_module import DynamicScatterVFE
from .basic.decoder import SimpleLinearDecoder
from .basic import cal_pose0to1

class SSF(nn.Module):
    def __init__(self, voxel_size = [0.2, 0.2, 6],
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3],
                 grid_feature_size = [512, 512],
                 decoder_option = "linear",
                 backbone_option = "simplesparse_unet",
                 num_iters = 4):
        super().__init__()
        self.voxel_layer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.voxel_encoder = DynamicScatterVFE(in_channels=3,
                                feat_channels=[32, 32],
                                voxel_size=voxel_size,
                                with_cluster_center=True,
                                with_voxel_center=True,
                                point_cloud_range=point_cloud_range,
                                norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
                                unique_once=True,)
        
        sparse_shape = grid_feature_size[::-1]
        self.backbone_option = backbone_option 
        if backbone_option == "simplesparse_unet":
            self.backbone = SimpleSparseUNet(in_channels=64,
                    sparse_shape=sparse_shape,
                    order=('conv', 'norm', 'act'),
                    norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
                    base_channels=64,
                    output_channels=128,
                    ndim=3,
                    encoder_channels=((128, ), (128, 128, ), (128, 128, ), (128, 128, 128), (256, 256, 256), (256, 256, 256)),
                    encoder_paddings=((1, ), (1, 1, ), (1, 1, ), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                    decoder_channels=((256, 256, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),
                    decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1)),)
        if decoder_option == "linear":
            self.head = SimpleLinearDecoder()

        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        transform_pc0s = []
        for batch_id in range(batch_sizes):
            selected_pc0 = batch["pc0"][batch_id]
            self.timer[0][0].start("pose")
            with torch.no_grad():
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
            transform_pc0s.append(transform_pc0)

        pc0s = torch.stack(transform_pc0s, dim=0)
        pc0s = torch.cat((pc0s, torch.ones((pc0s.size(0), pc0s.size(1),1)).to(pc0s.device) * 0.), dim=2)
        # indicator_pc0s = torch.ones((pc0s.size(0), pc0s.size(1),1), dtype=torch.int).to(pc0s.device) * 0
        pc1s = batch["pc1"]
        pc1s = torch.cat((pc1s, torch.ones((pc1s.size(0), pc1s.size(1),1)).to(pc1s.device) * 1.), dim=2)
        # indicator_pc1s = torch.ones((pc1s.size(0), pc1s.size(1),1), dtype=torch.int).to(pc1s.device) * 1
        pcs_concat = torch.cat((pc0s, pc1s), dim=1)
        self.timer[0].stop()

        self.timer[1].start("Voxelization")
        voxel_infos_dict = self.voxel_layer._concatenate_batch_results(self.voxel_layer(pcs_concat))
        pc0_voxel_features, pc0_voxel_coors, pc0_voxel2point_inds, pc0_dummy_mask = self.voxel_encoder(voxel_infos_dict['points'].clone(), voxel_infos_dict['voxel_coords'].clone(), indicator=0, return_inv=True)
        pc1_voxel_features, pc1_voxel_coors, pc1_voxel2point_inds, pc1_dummy_mask = self.voxel_encoder(voxel_infos_dict['points'].clone(), voxel_infos_dict['voxel_coords'].clone(), indicator=1, return_inv=True)
        
        voxel_infos_dict['indicator'] = voxel_infos_dict['points'][:,-1].to(torch.int)
        voxel_infos_dict['points'] = voxel_infos_dict['points'][:,:-1]
        
        self.timer[1].stop()

        self.timer[2].start("Encoder")
        if self.backbone_option == "simplesparse_unet":
            cat_voxel_info = dict()
            cat_voxel_info.update(
                voxel_feats = torch.cat((pc0_voxel_features, pc1_voxel_features), dim=1),
                voxel_coors = pc0_voxel_coors,
            )
            processed_info = self.backbone(cat_voxel_info)[0]
        elif self.backbone_option == "sparse_fastflow3d_unet":
            processed_info = self.backbone(pc0_voxel_features, pc0_voxel_coors, pc1_voxel_features, pc1_voxel_coors)[0]
        
        self.timer[2].stop()

        self.timer[3].start("Decoder")
        # pc0_pointwise_processed_feats = processed_info['voxel_feats'][pc0_voxel2point_inds][~pc0_dummy_mask]
        # pc0_pointwise_voxel_feats = pc0_voxel_features[pc0_voxel2point_inds][~pc0_dummy_mask]
        # pc0_batch_idx = pc0_voxel_coors[pc0_voxel2point_inds][:,0][~pc0_dummy_mask]
        pc0_point_offsets = voxel_infos_dict['point_offsets'][voxel_infos_dict['indicator']==0]

        # flows = self.head(pc0_pointwise_voxel_feats, pc0_pointwise_processed_feats, pc0_point_offsets, pc0_batch_idx)
        flows = self.head(torch.cat((pc0_voxel_features, pc1_voxel_features), dim=1),
            processed_info['voxel_feats'], pc0_point_offsets, pc0_voxel_coors, 
            pc0_voxel2point_inds, pc0_dummy_mask)
        self.timer[3].stop()

        # pc0_voxel_infos_lst, pc1_voxel_infos_lst = self.voxel_layer._split_results(voxel_infos_dict)
        pc0_voxel_infos_dict = dict()
        pc1_voxel_infos_dict = dict()
        indicator_mask_0 = voxel_infos_dict['indicator']==0
        indicator_mask_1 = voxel_infos_dict['indicator']==1
        for k, v in voxel_infos_dict.items():
            pc0_voxel_infos_dict[k] = v[indicator_mask_0]
            pc1_voxel_infos_dict[k] = v[indicator_mask_1]
        pc0_voxel_infos_lst = self.voxel_layer._split_batch_results(pc0_voxel_infos_dict)
        pc1_voxel_infos_lst = self.voxel_layer._split_batch_results(pc1_voxel_infos_dict)
        
        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        pc1_points_lst = [e["points"] for e in pc1_voxel_infos_lst]

        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        pc1_valid_point_idxes = [e["point_idxes"] for e in pc1_voxel_infos_lst]

        model_res = {
            "flow": flows,
            'pose_flow': pose_flows,

            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc0_points_lst": pc0_points_lst,
            
            "pc1_valid_point_idxes": pc1_valid_point_idxes,
            "pc1_points_lst": pc1_points_lst,

            'num_occupied_voxels': [processed_info['voxel_feats'].size(0)]
        }
        return model_res
