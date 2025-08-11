
"""
This file is directly copied from: 
https://github.com/dgist-cvlab/Flow4D

with slightly modification to have unified format with all benchmark.
"""

import torch.nn as nn
import dztimer, torch

from .basic import wrap_batch_pcs
from .basic.flow4d_module import DynamicEmbedder_4D
from .basic.flow4d_module import Network_4D, Seperate_to_3D, Point_head

class Flow4D(nn.Module):
    def __init__(self, voxel_size = [0.2, 0.2, 0.2],
                 point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size = [512, 512, 32],
                 num_frames = 5):
        super().__init__()

        point_output_ch = 16
        voxel_output_ch = 16

        self.num_frames = num_frames
        print('voxel_size = {}, pseudo_dims = {}, input_num_frames = {}'.format(voxel_size, grid_feature_size, self.num_frames))

        self.embedder_4D = DynamicEmbedder_4D(voxel_size=voxel_size,
                                        pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2], num_frames], 
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=point_output_ch)
        
        self.network_4D = Network_4D(in_channel=point_output_ch, out_channel=voxel_output_ch)
        self.seperate_feat = Seperate_to_3D(num_frames)
        self.pointhead_3D = Point_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch)
        
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
        pcs_dict = wrap_batch_pcs(batch, self.num_frames)
        self.timer[0].stop()

        self.timer[1].start("4D_voxelization")
        dict_4d = self.embedder_4D(pcs_dict)
        pc01_tesnor_4d = dict_4d['4d_tensor']
        pch1_3dvoxel_infos_lst = dict_4d['pch1_3dvoxel_infos_lst']
        pc0_3dvoxel_infos_lst =dict_4d['pc0_3dvoxel_infos_lst']

        pc0_point_feats_lst =dict_4d['pc0_point_feats_lst']
        pc0_num_voxels = dict_4d['pc0_num_voxels']

        pc1_3dvoxel_infos_lst =dict_4d['pc1_3dvoxel_infos_lst']
        self.timer[1].stop()

        self.timer[2].start("4D_backbone")
        pc_all_output_4d = self.network_4D(pc01_tesnor_4d) #all = past, current, next 다 합친것
        self.timer[2].stop()

        self.timer[3].start("4D pc01 to 3D pc0")
        pc0_last = self.seperate_feat(pc_all_output_4d)
        assert pc0_last.features.shape[0] == pc0_num_voxels, 'voxel number mismatch'
        self.timer[3].stop()

        self.timer[4].start("3D_sparsetensor_to_point and head")
        flows = self.pointhead_3D(pc0_last, pc0_3dvoxel_infos_lst, pc0_point_feats_lst)
        self.timer[4].stop()

        model_res = {
            "flow": flows, 
            'pose_flow': pcs_dict['pose_flows'], 

            "pc0_valid_point_idxes": [e["point_idxes"] for e in pc0_3dvoxel_infos_lst], 
            "pc0_points_lst": [e["points"] for e in pc0_3dvoxel_infos_lst] , 
            
            "pc1_valid_point_idxes": [e["point_idxes"] for e in pc1_3dvoxel_infos_lst],
            "pc1_points_lst": [e["points"] for e in pc1_3dvoxel_infos_lst],

            'pch1_valid_point_idxes': [e["point_idxes"] for e in pch1_3dvoxel_infos_lst] if pch1_3dvoxel_infos_lst != None else None,
        }
        return model_res