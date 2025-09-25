import torch
import torch.nn as nn
from typing import List, Dict

SPLIT_BATCH_SIZE = 512

class MMHeadDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 128)

        # FIXME: figure out how to set nheads and num_layers properly
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html
        transform_decoder_layers = nn.TransformerDecoderLayer(d_model=128, nhead=4)
        self.pts_off_transformer = nn.TransformerDecoder(transform_decoder_layers, num_layers=4)
        
        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*2, 32), nn.GELU(),
            nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T
        
        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)
        
        # [N, 128] [N, 128] -> [N, 1, 128]
        voxel_feature = concatenated_vectors.unsqueeze(1)
        point_offsets_feature = self.offset_encoder(point_offsets).unsqueeze(1)
        concatenated_feature = torch.zeros_like(voxel_feature)

        for spilt_range in range(0, concatenated_feature.shape[0], SPLIT_BATCH_SIZE):
            concatenated_feature[spilt_range:spilt_range+SPLIT_BATCH_SIZE] = self.pts_off_transformer(
                voxel_feature[spilt_range:spilt_range+SPLIT_BATCH_SIZE],
                point_offsets_feature[spilt_range:spilt_range+SPLIT_BATCH_SIZE]
            )
        
        flow = self.decoder(concatenated_feature.squeeze(1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results
    
class LinearDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 128)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*4, 32), nn.GELU(),
            nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T
        
        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)
        
        # [N, 3] -> [N, 128]
        point_offsets_feature = self.offset_encoder(point_offsets)

        flow = self.decoder(torch.cat([concatenated_vectors, point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results

class SimpleLinearDecoder(nn.Module):
    """
    Adapted from SST (https://github.com/tusen-ai/SST)
    Licensed under the Apache License 2.0.
    See https://www.apache.org/licenses/LICENSE-2.0
    """

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.offset_encoder = nn.Linear(3, pseudoimage_channels)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*4, 32), nn.GELU(),
            nn.Linear(32, 3))

    def forward_single(self, before_voxel_vectors: torch.Tensor,
                       after_voxel_vectors: torch.Tensor,
                       point_offsets: torch.Tensor,) -> torch.Tensor:

        # [N, 64] [N, 128] -> [N, 192]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)

        # [N, 3] -> [N, 64]
        point_offsets_feature = self.offset_encoder(point_offsets)

        flow = self.decoder(torch.cat([concatenated_vectors, point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            pc0_point_offsets: torch.Tensor, 
            pc0_voxel_coors: torch.Tensor, 
            pc0_voxel2point_inds: torch.Tensor, 
            pc0_dummy_mask: torch.Tensor,) -> List[torch.Tensor]:

        flow_results = []
        for idx in range(len(pc0_voxel_coors[:,0].unique())):
            voxel_batch_mask = pc0_voxel_coors[:,0]==idx
            pointwise_batch_mask = voxel_batch_mask[[pc0_voxel2point_inds]]
            realpoints_batch_mask = torch.logical_and(pointwise_batch_mask, ~pc0_dummy_mask)
            if before_pseudoimages.size(0) > pc0_point_offsets.size(0):
                a = torch.zeros((before_pseudoimages.size(0), 3)).to(before_pseudoimages.device)
                a[:pc0_point_offsets.size(0),:] = pc0_point_offsets
                pc0_point_offsets = a

            flow = self.forward_single(before_pseudoimages[pc0_voxel2point_inds][realpoints_batch_mask], 
                            after_pseudoimages[pc0_voxel2point_inds][realpoints_batch_mask], 
                            pc0_point_offsets[pc0_voxel2point_inds][realpoints_batch_mask])

            flow_results.append(flow)
        return flow_results


# from https://github.com/weiyithu/PV-RAFT/blob/main/model/update.py
class ConvGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convr = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convq = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        rh_x = torch.cat([r*h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h
    
class ConvGRUDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64, num_iters: int = 4):
        super().__init__()

        self.offset_encoder = nn.Linear(3, pseudoimage_channels)

        # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        self.gru = ConvGRU(input_dim=pseudoimage_channels, hidden_dim=pseudoimage_channels*2)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*3, pseudoimage_channels//2), nn.GELU(),
            nn.Linear(pseudoimage_channels//2, 3))
        self.num_iters = num_iters

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T
        
        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)
        
        # [N, 3] -> [N, 64]
        point_offsets_feature = self.offset_encoder(point_offsets)
        
        # [N, 128] -> [N, 128, 1]
        concatenated_vectors = concatenated_vectors.unsqueeze(2)

        for itr in range(self.num_iters):
            concatenated_vectors = self.gru(concatenated_vectors, point_offsets_feature.unsqueeze(2))

        flow = self.decoder(torch.cat([concatenated_vectors.squeeze(2), point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results


class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)

"""
Note(2024/7/18 21:11 Qingwen):
This is the decoder idea from DeFlow: https://github.com/KTH-RPL/DeFlow
If you use/find this helpful, please cite the respective publication as listed on the above website.
"""
class SparseGRUHead(ConvGRUDecoder):
    def __init__(self, voxel_feat_dim: int = 96, point_feat_dim: int = 32, num_iters = 2):
        super(SparseGRUHead, self).__init__(pseudoimage_channels=point_feat_dim, num_iters=num_iters)
        self.offset_encoder = None # otherwise DDP may find parameter not used in training.
        self.gru = ConvGRU(input_dim=point_feat_dim, hidden_dim=voxel_feat_dim)
        self.decoder = nn.Sequential(
            nn.Linear(voxel_feat_dim + point_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 3))
        
    def forward_single(self, voxel_feat, voxel_coords, point_offsets):
        # [N, voxel_feat_dim] -> [N, voxel_feat_dim, 1]
        concatenated_vectors = (voxel_feat[:, voxel_coords[:,2], voxel_coords[:,1], voxel_coords[:,0]].T).unsqueeze(2)
        for itr in range(self.num_iters):
            concatenated_vectors = self.gru(concatenated_vectors, point_offsets.unsqueeze(2))

        flow = self.decoder(torch.cat([concatenated_vectors.squeeze(2), point_offsets], dim=1))
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