"""
# Created: 2024-11-15 21:33
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of 
# * DeltaFlow (https://github.com/Kin-Zhang/DeltaFlow)
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True
from .encoder import DynamicVoxelizer, DynamicPillarFeatureNet
import dztimer

class SparseVoxelNet(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, decay_factor=1.0, timer=None) -> None:
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
        self.num_feature = feat_channels
        self.decay_factor = decay_factor
        if timer is None:
            self.timer = dztimer.Timing()
            self.timer.start("Total")
        else:
            self.timer = timer

    def process_batch(self, voxel_info_list, if_return_point_feats=False):
        voxel_feats_list_batch = []
        voxel_coors_list_batch = []
        point_feats_lst = []

        for batch_index, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            if if_return_point_feats:
                point_feats_lst.append(point_feats)
            batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
            voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)
            voxel_feats_list_batch.append(voxel_feats)
            voxel_coors_list_batch.append(voxel_coors_batch)

        voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0)
        coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32)
        
        if if_return_point_feats:
            return voxel_feats_sp, coors_batch_sp, point_feats_lst
        
        return voxel_feats_sp, coors_batch_sp
    
    def forward(self, input_dict) -> torch.Tensor:
        bz_ = len(input_dict['pc0s'])
        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pch')], reverse=True)
        frame_keys += ['pc0s']

        pc1_voxel_info_list = self.voxelizer(input_dict['pc1s'])
        pc1_voxel_feats_sp, pc1_coors_batch_sp = self.process_batch(pc1_voxel_info_list)
        pc1s_num_voxels = pc1_voxel_feats_sp.shape[0]
        sparse_max_size = [bz_, *self.voxel_spatial_shape, self.num_feature]
        sparse_pc1 = torch.sparse_coo_tensor(pc1_coors_batch_sp.t(), pc1_voxel_feats_sp, size=sparse_max_size)
        sparse_diff = torch.sparse_coo_tensor(pc1_coors_batch_sp.t(), pc1_voxel_feats_sp * 0.0, size=sparse_max_size)
        pch1s_3dvoxel_infos_lst = None
        pc0_point_feats_lst = []

        # (0, 'pch2s'), (1, 'pch1s'), (2, 'pc0s')
        # reversed: (0, 'pc0s'), (1, 'pch1s'), (2, 'pch2s')
        for time_index, frame_key in enumerate(reversed(frame_keys)): 
            self.timer[0].start("Point Feature Voxelize")
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)

            if frame_key == 'pc0s':
                voxel_feats_sp, coors_batch_sp, pc0_point_feats_lst = self.process_batch(voxel_info_list, if_return_point_feats=True)
            else:
                voxel_feats_sp, coors_batch_sp = self.process_batch(voxel_info_list)

            sparse_pcx = torch.sparse_coo_tensor(coors_batch_sp.t(), voxel_feats_sp, size=sparse_max_size)
            sparse_diff = sparse_diff + (sparse_pc1 - sparse_pcx) * pow(self.decay_factor, time_index)
            self.timer[0].stop()

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]
            elif frame_key == 'pch1s':
                pch1s_3dvoxel_infos_lst = voxel_info_list

        self.timer[2].start("D_Delta_Sparse")
        features = sparse_diff.coalesce().values() / (time_index + 1)
        indices = sparse_diff.coalesce().indices().t().to(dtype=torch.int32)
        all_pcdiff_sparse = spconv.SparseConvTensor(features.contiguous(), indices.contiguous(), self.voxel_spatial_shape, bz_)
        self.timer[2].stop()

        output = {
            'delta_sparse': all_pcdiff_sparse,
            'pch1_3dvoxel_infos_lst': pch1s_3dvoxel_infos_lst,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_num_voxels': pc0s_num_voxels,
            'pc1_3dvoxel_infos_lst': pc1_voxel_info_list,
            'pc1_num_voxels': pc1s_num_voxels,
            'd_num_voxels': indices.shape[0]
        }
        return output

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=0, indice_key=None):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(inc, outc, kernel_size=ks, stride=stride, dilation=dilation, padding=padding, bias=False, \
                              indice_key=indice_key, algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, indice_key, ks=3):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseInverseConv3d(inc, outc, kernel_size=ks, indice_key=indice_key, bias=False, algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=0):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inc, outc, kernel_size=ks, stride=stride, dilation=dilation, padding=padding, bias=False, \
                                algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(outc, outc, kernel_size=ks, stride=stride, dilation=dilation, padding=padding, bias=False, \
                                algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc)
        )

        if inc == (outc * self.expansion) and stride == 1:
            self.downsample = None
        else:
            self.downsample = spconv.SparseSequential(
                spconv.SubMConv3d(inc, outc, kernel_size=1, dilation=1,
                                stride=stride, algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(outc)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x.features
        out = self.net(x)
        if self.downsample is not None:
            identity = self.downsample(x).features
        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(self.relu(out.features))

        return out
    
'''
Reference when I wrote MinkUNet:
* https://github.com/PJLab-ADG/OpenPCSeg/blob/master/pcseg/model/segmentor/voxel/minkunet/minkunet.py
* https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/backbones/minkunet_backbone.py
* https://github.com/mit-han-lab/spvnas/blob/master/core/models/semantic_kitti/minkunet.py
'''
class MinkUNet(nn.Module):
    def __init__(self, 
                 cs=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16], 
                 num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2]):
        super().__init__()
        
        inc = cs[0]
        cs = cs[1:] # remove the first input channel after conv_input
        self.block = ResidualBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(inc, cs[0], kernel_size=3, stride=1, padding=1, bias=False, \
                                indice_key="subm0", algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True),

            spconv.SubMConv3d(cs[0], cs[0], kernel_size=3, stride=1, padding=1, bias=False, \
                                indice_key="subm0", algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True)
        )
        self.in_channels = cs[0]

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm1"),
            *self._make_layer(self.block, cs[1], num_layer[0])
        )
        # inside every make_layer: self.in_channels = out_channels * block.expansion
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm2"),
            *self._make_layer(self.block, cs[2], num_layer[1])
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm3"),
            *self._make_layer(self.block, cs[3], num_layer[2])
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm4"),
            *self._make_layer(self.block, cs[4], num_layer[3])
        )

        self.up1 = [BasicDeconvolutionBlock(self.in_channels, cs[5], ks=2, indice_key="subm4")]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(nn.Sequential(*self._make_layer(self.block, cs[5], num_layer[4])))
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [BasicDeconvolutionBlock(cs[5], cs[6], ks=2, indice_key="subm3")]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(nn.Sequential(*self._make_layer(self.block, cs[6], num_layer[5])))
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [BasicDeconvolutionBlock(cs[6], cs[7], ks=2, indice_key="subm2")]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(nn.Sequential(*self._make_layer(self.block, cs[7], num_layer[6])))
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [BasicDeconvolutionBlock(cs[7], cs[8], ks=2, indice_key="subm1")]
        self.in_channels = cs[8] + cs[0] * self.block.expansion
        self.up4.append(nn.Sequential(*self._make_layer(self.block, cs[8], num_layer[7])))
        self.up4 = nn.ModuleList(self.up4)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride=stride)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(self.in_channels, out_channels)
            )
        return layers
    
    def forward(self, x):
        x = self.conv_input(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = y1.replace_feature(torch.cat([y1.features, x3.features], dim=1))
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = y2.replace_feature(torch.cat([y2.features, x2.features], dim=1))
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = y3.replace_feature(torch.cat([y3.features, x1.features], dim=1))
        y3 = self.up3[1](y3) # Dense shape: [B, C, X, Y, Z]; [B, 32, 256, 256, 16]
        
        y4 = self.up4[0](y3)
        y4 = y4.replace_feature(torch.cat([y4.features, x.features], dim=1))
        y4 = self.up4[1](y4)

        return y4