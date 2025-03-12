import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
from assets.cuda.mmdet import build_norm_layer
from assets.cuda.mmcv import DynamicScatter

def scatter_v2(feat, coors, mode, return_inv=True, min_points=0, unq_inv=None, new_coors=None):
    """
    Adapted from SST (https://github.com/tusen-ai/SST)
    Licensed under the Apache License 2.0.
    See https://www.apache.org/licenses/LICENSE-2.0
    Args:
        feat (torch.Tensor): The feature tensor to be scattered.
        coors (torch.Tensor): The coordinates tensor corresponding to the features.
        mode (str): The reduction mode to apply. Options are 'avg' (alias for 'mean'), 'max', 'mean', and 'sum'.
        return_inv (bool, optional): Whether to return the inverse indices. Default is True.
        min_points (int, optional): Minimum number of points required to keep a coordinate. Default is 0.
        unq_inv (torch.Tensor, optional): Precomputed unique inverse indices. Default is None.
        new_coors (torch.Tensor, optional): Precomputed new coordinates. Default is None.
    Returns:
        tuple: A tuple containing:
            - new_feat (torch.Tensor): The scattered feature tensor.
            - new_coors (torch.Tensor): The new coordinates tensor.
            - unq_inv (torch.Tensor, optional): The unique inverse indices tensor, if return_inv is True.
    Raises:
        AssertionError: If the size of feat and coors do not match.
        NotImplementedError: If the mode is not one of 'max', 'mean', or 'sum'.
    """
    assert feat.size(0) == coors.size(0)
    if mode == 'avg':
        mode = 'mean'


    if unq_inv is None and min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    elif unq_inv is None:
        new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
    else:
        assert new_coors is not None, 'please pass new_coors for interface consistency, caller: {}'.format(traceback.extract_stack()[-2][2])


    if min_points > 0:
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
        feat = feat[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

    if mode == 'max':
        new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
    elif mode in ('mean', 'sum'):
        new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
    else:
        raise NotImplementedError

    if not return_inv:
        return new_feat, new_coors
    else:
        return new_feat, new_coors, unq_inv

class DynamicVFELayer(nn.Module):
    """
    Adapted from SST (https://github.com/tusen-ai/SST)
    Licensed under the Apache License 2.0.
    See https://www.apache.org/licenses/LICENSE-2.0
    Replace the Voxel Feature Encoder layer in VFE layers.
    This layer has the same utility as VFELayer above
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)
                 ):
        super(DynamicVFELayer, self).__init__()
        self.fp16_enabled = False
        # self.units = int(out_channels / 2)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    # @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (torch.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.
        Returns:
            torch.Tensor: point features in shape (M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = F.relu(x)
        return pointwise

class DynamicVFE(nn.Module):
    """
    Adapted from SST (https://github.com/tusen-ai/SST)
    Licensed under the Apache License 2.0.
    See https://www.apache.org/licenses/LICENSE-2.0
    Dynamic Voxel feature encoder used in DV-SECOND.
    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.
    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 ):
        super(DynamicVFE, self).__init__()
        # assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        # if fusion_layer is not None:
        #     self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.
        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels
        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    # @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.
        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.
        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        # List of points and coordinates for each batch
        voxel_info_list = self.voxelizer(points)

        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors

class DynamicScatterVFE(DynamicVFE):
    """ 
    Originally from SST (https://github.com/tusen-ai/SST)
    Modified by Ajinkya Khoche (https://ajinkyakhoche.github.io/) for SSF (https://github.com/KTH-RPL/SSF)
    Same with DynamicVFE but use torch_scatter to avoid construct canvas in map_voxel_center_to_point.
    The canvas is very memory-consuming when use tiny voxel size (5cm * 5cm * 5cm) in large 3D space.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 unique_once=False,
                 ):
        super(DynamicScatterVFE, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.unique_once = unique_once
        # self.voxel_layer = DynamicVoxelizer(voxel_size=voxel_size,
        #                                   point_cloud_range=point_cloud_range)
    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    # @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                indicator=None,
                img_feats=None,
                img_metas=None,
                return_inv=False):

        if self.unique_once:
            new_coors, unq_inv_once = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv_once = None

        if features.size(1) > 3:
            indicator_mask = features[:,-1] == indicator

            coors_mask = torch.zeros((new_coors.size(0)), dtype=torch.bool).to(features.device) 
            coors_mask[unq_inv_once[indicator_mask].unique()] = True
            df_x = new_coors[~coors_mask][:,3] * self.vx + self.x_offset
            df_y = new_coors[~coors_mask][:,2] * self.vy + self.y_offset
            df_z = new_coors[~coors_mask][:,1] * self.vz + self.z_offset
            dummy_features = torch.cat((df_x[:,None], df_y[:,None], df_z[:,None]), dim=1)

            features = features[indicator_mask, :3]
            coors = coors[indicator_mask, :]
            unq_inv_once = unq_inv_once[indicator_mask]

            # append dummy features. make sure the ordering of new_coors is preserved!
            features = torch.cat((dummy_features, features), dim=0)
            coors = torch.cat((new_coors[~coors_mask], coors), dim=0)
            unq_inv_once = torch.cat((torch.where(coors_mask==False)[0], unq_inv_once), dim=0)
            dummy_mask = torch.zeros((features.size(0)), dtype=torch.bool).to(features.device)
            dummy_mask[:dummy_features.size(0)] = True
        else:
            dummy_mask = torch.zeros((features.size(0)), dtype=torch.bool).to(features.device)

        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, _, unq_inv = scatter_v2(features, coors, mode='avg', new_coors=new_coors, unq_inv=unq_inv_once)
            points_mean = self.map_voxel_center_to_point(voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster / self.rel_dist_scaler)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors, unq_inv = scatter_v2(point_feats, coors, mode=self.mode, new_coors=new_coors, unq_inv=unq_inv_once)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats

        # Suppress dummy voxel features 
        voxel_feats[unq_inv[dummy_mask]] = 0.

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv, dummy_mask
        else:
            return voxel_feats, voxel_coors
