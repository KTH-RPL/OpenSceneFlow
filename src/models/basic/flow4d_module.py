"""
# Updated: 2024-07-12 01:16
# Please visit: https://github.com/dgist-cvlab/Flow4D to cite this network model.
# Author: Jaeyeul Kim (jykim94@dgist.ac.kr), Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# 
Changelog:
2024/07/13 (Qingwen): spconv.ConvAlgo.Native needed to be added to all spconv layers, add some references here:
- https://github.com/traveller59/spconv/issues/467
- https://github.com/traveller59/spconv/issues/482

I tried half hour to find the solution... and note down here for other readers:
1. It's because of <return ALL_TENSOR_OP_MAP[(shape_tuple, dtype_ab, dtype_c)]> KeyError: ((16, 8, 8), float, float)
2. Then I checked cumm/gemm file: # ((16, 8, 8), dtypes.float32, dtypes.float32): MmaM16N8K8F32((8, 0)),
3. This line is commented that's why float32 float 32 is KeyError
4. And I didn't find a solution to have cumm/gemm work out. But change all algorithm on layers to: spconv.ConvAlgo.Native
5. 2025/03/12 (Qingwen): Another way to solve it is limit `spconv-cu117==2.3.6` and make sure you didn't have any version in local python environment.

This file is originally copied from: https://github.com/dgist-cvlab/Flow4D
with some modifications to have unified format with all benchmark. Check above changelog I made.

"""
import torch, os
import torch.nn as nn

from .encoder import DynamicVoxelizer, DynamicPillarFeatureNet
import spconv.pytorch as spconv
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

class DynamicEmbedder_4D(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        
        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, input_dict) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pch')], reverse=True)
        frame_keys += ['pc0s', 'pc1s']
        
        pch1s_3dvoxel_infos_lst = None
        pc0_point_feats_lst, pc1_point_feats_lst = [], []
        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)

            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)

                if frame_key == 'pc0s':
                    pc0_point_feats_lst.append(point_feats)
                elif frame_key == 'pc1s':
                    pc1_point_feats_lst.append(point_feats)
                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0)
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32)
            
            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)

            voxel_feats_list.append(voxel_feats_sp)
            voxel_coors_list.append(coors_batch_sp_4d)

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]
            elif frame_key == 'pc1s':
                pc1s_3dvoxel_infos_lst = voxel_info_list
                pc1s_num_voxels = voxel_feats_sp.shape[0]
            elif frame_key == 'pch1s':
                pch1s_3dvoxel_infos_lst = voxel_info_list

        all_voxel_feats_sp = torch.cat(voxel_feats_list, dim=0)
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list, dim=0)

        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(), self.voxel_spatial_shape, int(batch_index + 1))
        # dense shape: B C X Y Z T
        output = {
            '4d_tensor': sparse_tensor_4d,
            'pch1_3dvoxel_infos_lst': pch1s_3dvoxel_infos_lst,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_num_voxels': pc0s_num_voxels,
            'pc1_3dvoxel_infos_lst': pc1s_3dvoxel_infos_lst,
            'pc1_point_feats_lst': pc1_point_feats_lst,
            'pc1_num_voxels': pc1s_num_voxels
        }

        return output
    
def conv1x1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1,1,1,3), stride=stride,
                             padding=(0,0,0,1), bias=False, indice_key=indice_key)

def conv3x3x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(3,3,3,1), stride=stride,
                             padding=(1,1,1,0), bias=False, indice_key=indice_key)

def conv1x1x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1,1,1,1), stride=stride,
                             padding=0, bias=False, indice_key=indice_key)


def conv3x3x3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(3,3,3,3), stride=stride,
                             padding=(1,1,1,1), bias=False, indice_key=indice_key)


class Seperate_to_3D(nn.Module):
    def __init__(self, num_frames):
        super(Seperate_to_3D, self).__init__()
        self.num_frames = num_frames
        #self.return_pc1 = return_pc1

    def forward(self, sparse_4D_tensor):

        indices_4d = sparse_4D_tensor.indices
        features_4d = sparse_4D_tensor.features
        
        pc0_time_value = self.num_frames-2

        mask_pc0 = (indices_4d[:, -1] == pc0_time_value)
        
        pc0_indices = indices_4d[mask_pc0][:, :-1] 
        pc0_features = features_4d[mask_pc0]

        pc0_sparse_3D = sparse_4D_tensor.replace_feature(pc0_features)
        pc0_sparse_3D.spatial_shape = sparse_4D_tensor.spatial_shape[:-1]
        pc0_sparse_3D.indices = pc0_indices

        return pc0_sparse_3D

class SpatioTemporal_Decomposition_Block(nn.Module):
    def __init__(self, in_filters, mid_filters, out_filters, indice_key=None, down_key = None, pooling=False, z_pooling=True, interact=False):
        super(SpatioTemporal_Decomposition_Block, self).__init__()


        self.pooling = pooling

        self.act = nn.LeakyReLU()

        self.spatial_conv_1 = conv3x3x3x1(in_filters, mid_filters, indice_key=indice_key + "bef")
        self.bn_s_1 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_1 = conv1x1x1x3(in_filters, mid_filters)
        self.bn_t_1 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_1 = conv1x1x1x1(mid_filters*2+in_filters, mid_filters, indice_key=indice_key + "1D")
        self.bn_fusion_1 = nn.BatchNorm1d(mid_filters)


        self.spatial_conv_2 = conv3x3x3x1(mid_filters, mid_filters, indice_key=indice_key + "bef")
        self.bn_s_2 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_2 = conv1x1x1x3(mid_filters, mid_filters)
        self.bn_t_2 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_2 = conv1x1x1x1(mid_filters*3, out_filters, indice_key=indice_key + "1D")
        self.bn_fusion_2 = nn.BatchNorm1d(out_filters)


        if self.pooling:
            if z_pooling == True:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2,2,2,1), stride=(2,2,2,1), indice_key=down_key, bias=False)
            else:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2,2,1,1), stride=(2,2,1,1), indice_key=down_key, bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        #ST block
        S_feat_1 = self.spatial_conv_1(x)
        S_feat_1 = S_feat_1.replace_feature(self.bn_s_1(S_feat_1.features))
        S_feat_1 = S_feat_1.replace_feature(self.act(S_feat_1.features))

        T_feat_1 = self.temporal_conv_1(x)
        T_feat_1 = T_feat_1.replace_feature(self.bn_t_1(T_feat_1.features))
        T_feat_1 = T_feat_1.replace_feature(self.act(T_feat_1.features))

        ST_feat_1 = x.replace_feature(torch.cat([S_feat_1.features, T_feat_1.features, x.features], 1)) #residual까지 concate

        ST_feat_1 = self.fusion_conv_1(ST_feat_1)
        ST_feat_1 = ST_feat_1.replace_feature(self.bn_fusion_1(ST_feat_1.features))
        ST_feat_1 = ST_feat_1.replace_feature(self.act(ST_feat_1.features))

        #TS block
        S_feat_2 = self.spatial_conv_2(ST_feat_1)
        S_feat_2 = S_feat_2.replace_feature(self.bn_s_2(S_feat_2.features))
        S_feat_2 = S_feat_2.replace_feature(self.act(S_feat_2.features))

        T_feat_2 = self.temporal_conv_2(ST_feat_1)
        T_feat_2 = T_feat_2.replace_feature(self.bn_t_2(T_feat_2.features))
        T_feat_2 = T_feat_2.replace_feature(self.act(T_feat_2.features))

        ST_feat_2 = x.replace_feature(torch.cat([S_feat_2.features, T_feat_2.features, ST_feat_1.features], 1)) #residual까지 concate
        
        ST_feat_2 = self.fusion_conv_2(ST_feat_2)
        ST_feat_2 = ST_feat_2.replace_feature(self.bn_fusion_2(ST_feat_2.features))
        ST_feat_2 = ST_feat_2.replace_feature(self.act(ST_feat_2.features))

        if self.pooling: 
            pooled = self.pool(ST_feat_2)
            return pooled, ST_feat_2
        else:
            return ST_feat_2


class Network_4D(nn.Module):
    def __init__(self, in_channel=16, out_channel=16, model_size = 16):
        super().__init__()

        SpatioTemporal_Block = SpatioTemporal_Decomposition_Block

        self.model_size = model_size
        

        self.STDB_1_1_1 = SpatioTemporal_Block(in_channel, model_size, model_size, indice_key="st1_1", down_key='floor1')
        self.STDB_1_1_2 = SpatioTemporal_Block(model_size, model_size, model_size*2, indice_key="st1_1", down_key='floor1', pooling=True) #512 512 32 -> 256 256 16

        self.STDB_2_1_1 = SpatioTemporal_Block(model_size*2, model_size*2, model_size*2, indice_key="st2_1", down_key='floor2')
        self.STDB_2_1_2 = SpatioTemporal_Block(model_size*2, model_size*2, model_size*4, indice_key="st2_1", down_key='floor2', pooling=True) #256 256 16 -> 128 128 8

        self.STDB_3_1_1 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st3_1", down_key='floor3')
        self.STDB_3_1_2 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st3_1", down_key='floor3', pooling=True) #128 128 8 -> 64 64 4

        self.STDB_4_1_1 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st4_1", down_key='floor4')
        self.STDB_4_1_2 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st4_1", down_key='floor4', pooling=True, z_pooling=False) #64 64 4 -> 64 64 4

        self.STDB_5_1_1 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st5_1")
        self.STDB_5_1_2 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st5_1")
        self.up_subm_5 = spconv.SparseInverseConv4d(model_size*4, model_size*4, kernel_size=(2,2,1,1), indice_key='floor4', bias=False) #zpooling false

        self.STDB_4_2_1 = SpatioTemporal_Block(model_size*8, model_size*8, model_size*4, indice_key="st4_2")
        self.up_subm_4 = spconv.SparseInverseConv4d(model_size*4, model_size*4, kernel_size=(2,2,2,1), indice_key='floor3', bias=False)

        self.STDB_3_2_1 = SpatioTemporal_Block(model_size*8, model_size*8, model_size*4, indice_key="st3_2")
        self.up_subm_3 = spconv.SparseInverseConv4d(model_size*4, model_size*4, kernel_size=(2,2,2,1), indice_key='floor2', bias=False)

        self.STDB_2_2_1 = SpatioTemporal_Block(model_size*8, model_size*4, model_size*4, indice_key="st_2_2")
        self.up_subm_2 = spconv.SparseInverseConv4d(model_size*4, model_size*2, kernel_size=(2,2,2,1), indice_key='floor1', bias=False)

        self.STDB_1_2_1 = SpatioTemporal_Block(model_size*4, model_size*2, out_channel, indice_key="st_1_2")
        

    def forward(self, sp_tensor):

        sp_tensor = self.STDB_1_1_1(sp_tensor) # [B, C, 512, 512, 32, T] [512,512,32] = [coor_x, coor_y, coor_z]
        down_2, skip_1 = self.STDB_1_1_2(sp_tensor)

        down_2 = self.STDB_2_1_1(down_2)
        down_3, skip_2 = self.STDB_2_1_2(down_2)

        down_3 = self.STDB_3_1_1(down_3)
        down_4, skip_3 = self.STDB_3_1_2(down_3)

        down_4 = self.STDB_4_1_1(down_4)
        down_5, skip_4 = self.STDB_4_1_2(down_4)

        down_5 = self.STDB_5_1_1(down_5)
        down_5 = self.STDB_5_1_2(down_5)

        up_4 = self.up_subm_5(down_5)
        up_4 = up_4.replace_feature(torch.cat((up_4.features, skip_4.features), 1))
        up_4 = self.STDB_4_2_1(up_4)

        up_3 = self.up_subm_4(up_4)
        up_3 = up_3.replace_feature(torch.cat((up_3.features, skip_3.features), 1))
        up_3 = self.STDB_3_2_1(up_3)

        up_2 = self.up_subm_3(up_3)
        up_2 = up_2.replace_feature(torch.cat((up_2.features, skip_2.features), 1))
        up_2 = self.STDB_2_2_1(up_2)

        up_1 = self.up_subm_2(up_2)
        up_1 = up_1.replace_feature(torch.cat((up_1.features, skip_1.features), 1))
        up_1 = self.STDB_1_2_1(up_1)

        return up_1
    
class Point_head(nn.Module):
    def __init__(self, voxel_feat_dim: int = 96, point_feat_dim: int = 32):
        super().__init__()

        self.input_dim = voxel_feat_dim + point_feat_dim

        self.PPmodel_flow = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 3)
        )

    def forward_single(self, voxel_feat, voxel_coords, point_feat):

        voxel_to_point_feat = voxel_feat[:, voxel_coords[:,2], voxel_coords[:,1], voxel_coords[:,0]].T 
        concated_point_feat = torch.cat([voxel_to_point_feat, point_feat],dim=-1)

        flow = self.PPmodel_flow(concated_point_feat)

        return flow

    def forward(self, sparse_tensor, voxelizer_infos, pc0_point_feats_lst): 
        
        voxel_feats = sparse_tensor.dense()

        flow_outputs = []
        batch_idx = 0
        for voxelizer_info in voxelizer_infos:
            voxel_coords = voxelizer_info["voxel_coords"]
            point_feat = pc0_point_feats_lst[batch_idx]
            voxel_feat = voxel_feats[batch_idx, :]
            flow = self.forward_single(voxel_feat, voxel_coords, point_feat)
            batch_idx += 1 
            flow_outputs.append(flow)

        return flow_outputs