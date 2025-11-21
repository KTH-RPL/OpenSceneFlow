
"""
# Created: 2024-11-16 14:31
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of 
# * DeltaFlow (https://github.com/Kin-Zhang/DeltaFlow)
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import torch.nn as nn
import dztimer, torch

from .basic import wrap_batch_pcs
from .basic.sparse_encoder import MinkUNet, SparseVoxelNet
from .basic.decoder import SparseGRUHead
from .basic.flow4d_module import Point_head

class DeltaFlow(nn.Module):
    def __init__(self, voxel_size = [0.2, 0.2, 0.2],
                 point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size = [512, 512, 32],
                 num_frames = 2,
                 planes = [16, 32, 64, 128, 256, 256, 128, 64, 32, 16], 
                 num_layer = [2,  2,  2,   2,   2,   2,   2,  2,  2],
                 decay_factor = 1.0,
                 decoder_option = "default", 
                 ):
        super().__init__()
        # NOTE(Qingwen) [0]: point feat input channel, [-1]: voxel feat output channel
        point_output_ch = planes[0]
        voxel_output_ch = planes[-1]
        self.timer = dztimer.Timing()
        self.num_frames = num_frames
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print('[LOG] Param detail: voxel_size = {}, pseudo_dims = {}, num_frames={}'.format(voxel_size, grid_feature_size, num_frames))
            print('[LOG] Model detail: planes = {}, time decay = {}, decoder = {}'.format(planes, decay_factor, decoder_option))
        
        self.pc2voxel = SparseVoxelNet(voxel_size=voxel_size,
                                pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2]], 
                                point_cloud_range=point_cloud_range,
                                feat_channels=point_output_ch, decay_factor=decay_factor, timer=self.timer[1])
        self.backbone = MinkUNet(planes, num_layer)
        if decoder_option == "deflow":
            self.flowdecoder = SparseGRUHead(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch, num_iters=1)
        else:
            self.flowdecoder = Point_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch)
        
        self.voxel_spatial_shape = grid_feature_size
        self.cnt = 0
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
               * batch input detail: [pc0, pc1, pose0, pose1]
                    - based on the num_frames input, we may have pch1, ... etc for past frames
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        self.cnt += 1
        self.timer[0].start("Data Preprocess")
        pcs_dict = wrap_batch_pcs(batch, num_frames=self.num_frames)
        self.timer[0].stop()

        self.timer[1].start("3D Sparse Voxel")
        sparse_dict = self.pc2voxel(pcs_dict)
        self.timer[1].stop()

        self.timer[3].start("3D Network")
        backbone_res = self.backbone(sparse_dict['delta_sparse'])
        pc0_3dvoxel_infos_lst = sparse_dict['pc0_3dvoxel_infos_lst']
        pc1_3dvoxel_infos_lst = sparse_dict['pc1_3dvoxel_infos_lst']
        self.timer[3].stop()

        self.timer[4].start("Flow Decoder")
        flows = self.flowdecoder(backbone_res, pc0_3dvoxel_infos_lst, sparse_dict['pc0_point_feats_lst'])
        self.timer[4].stop()

        model_res = {
            "d_num_voxels": [sparse_dict['d_num_voxels']],
            "flow": flows, # relative flow w.o pose flow (ego motion)
            'pose_flow': pcs_dict['pose_flows'], 

            "pc0_valid_point_idxes": [e["point_idxes"] for e in pc0_3dvoxel_infos_lst], 
            "pc0_points_lst": [e["points"] for e in pc0_3dvoxel_infos_lst] , 
            
            "pc1_valid_point_idxes": [e["point_idxes"] for e in pc1_3dvoxel_infos_lst],
            "pc1_points_lst": [e["points"] for e in pc1_3dvoxel_infos_lst],
        }
        if 'pch1_3dvoxel_infos_lst' in sparse_dict and sparse_dict['pch1_3dvoxel_infos_lst'] is not None:
            model_res["pch1_valid_point_idxes"] = [e["point_idxes"] for e in sparse_dict['pch1_3dvoxel_infos_lst']]
            model_res["pch1_points_lst"] = [e["points"] for e in sparse_dict['pch1_3dvoxel_infos_lst']]
        else:
            model_res["pch1_valid_point_idxes"] = None
            
        return model_res