import numpy as np
import random
import glob
import warnings
import parmap
import os
import itertools
import shutil
import argparse
import copy
import sys
import yaml
import json
import gc
import datetime, time
import open3d as o3d
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
import plotly
from tqdm import tqdm
import torch
import torch.nn as nn
import pytorch3d.ops as pytorch3d_ops
import plotly.graph_objs as go
from util_model import Decoder
from util_loss import warped_pc_loss
from utils_visualization import visualize_pcd
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
# add cuda implementation
from ht.ht_cuda import HT_CUDA

# this loads a pair of frames and gt flow. Note 1). the source has been compensated by ego motion; and 2) ground has been removed.
def dataloader_minimal(data_path):
    data = np.load(data_path)
    pcl_0 = data['pc1']
    pcl_1 = data['pc2']
    valid_0 = data['pc1_flows_valid_idx']
    valid_1 = data['pc2_flows_valid_idx']
    flow_0_1 = data['gt_flow_0_1']
    class_0 = data['pc1_classes']
    class_1 = data['pc2_classes']

    pcl_0 = pcl_0[valid_0]
    pcl_1 = pcl_1[valid_1]
    flow_0_1 = flow_0_1[valid_0]
    class_0 = class_0[valid_0]
    class_1 = class_1[valid_1]
    # print('pc range: ',
    #     pcl_0[:, 0].min(), pcl_0[:, 0].max(), 
    #     pcl_0[:, 1].min(), pcl_0[:, 1].max(), 
    #     pcl_1[:, 0].min(), pcl_1[:, 0].max(), 
    #     pcl_1[:, 1].min(), pcl_1[:, 1].max(), 
    # )
    # print('class 0: ', np.unique(class_0))
    # visualize_pcd(
    #     np.concatenate([pcl_0, pcl_1, pcl_0+flow_0_1], axis=0),
    #     np.concatenate([np.zeros(len(pcl_0))+1, np.zeros(len(pcl_1))+2, np.zeros(len(pcl_0))+0], axis=0),
    #     num_colors=3, 
    #     title=f'visualize input: src-g, dst-b, src+flow-r: {data_path}'
    #     )
    data_dict = {
        'point_src': pcl_0, 
        'point_dst': pcl_1, 
        'scene_flow': flow_0_1,
        'data_path': data_path
    }
    return data_dict

if __name__ == "__main__":

    # Initialization
    random.seed(0)
    np.random.seed(0)
    # fix open3d seed. question mark there: sometimes o3d outputs different results.
    o3d.utility.random.seed(0) # only in open3d>=0.16
    multiprocessing.set_start_method('forkserver') # or 'spawn'

    torch.manual_seed(0)
    if torch.cuda.is_available():
        assert torch.cuda.device_count()==1
        device_name = f'cuda:{torch.cuda.current_device()}'
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        torch.cuda.empty_cache()
        gc.collect()
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print(f"Let's use {torch.cuda.device_count()}, {torch.cuda.get_device_name()} GPU(s)!")
    else:
        print("CUDA is not available")
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f'device: {device}')

    model = Decoder().to(device)
    b = 4
    l = 40000
    m = 9 # neighbors in src
    n =16 # neighbors in dst
    vote = HT_CUDA(m, n, 3.3, 3.3, 0.1, 0.1).to(device)

    files = glob.glob(os.path.join('demo.npz'))
    print('total files: ', len(files))
    for file in files:
        data = dataloader_minimal(file)
        points_src = data['point_src']
        points_dst = data['point_dst']

        # random downsampling
        points_src = points_src[np.random.choice(np.arange(0, len(points_src)), l, replace=False)]
        points_dst = points_dst[np.random.choice(np.arange(0, len(points_src)), l, replace=False)]

        # coordinates
        points_src = torch.from_numpy(points_src).float().to(device)
        points_dst = torch.from_numpy(points_dst).float().to(device)
        print('points: ', points_src.shape, points_dst.shape)

        # feats. this is a minimal exmaple. in pratice, they are learned from some backbone model; 
        feats_src = torch.randn((len(points_src), 64)).float().to(device)
        feats_dst = torch.randn((len(points_dst), 64)).float().to(device)
        feats_src /= torch.linalg.norm(feats_src, dim=-1, keepdims=True)
        feats_dst /= torch.linalg.norm(feats_dst, dim=-1, keepdims=True)
        print('feats: ', feats_src.shape, feats_dst.shape)

        # batch, think about how to batch point clouds of various length
        points_src, points_dst = points_src[None].expand(b, l, 3).contiguous(), points_dst[None].expand(b, l, 3).contiguous()
        feats_src, feats_dst = feats_src[None].expand(b, l, 64).contiguous(), feats_dst[None].expand(b, l, 64).contiguous()

        # only need to calculate once per batch
        with torch.no_grad():

            # process coordinates
            # knn of src in dst
            knn_dst = pytorch3d_ops.knn_points(points_src, points_dst, lengths1=None, lengths2=None, K=16, return_nn=True, return_sorted=False)
            dists_dst, idxs_dst, pts_dst = knn_dst[0], knn_dst[1], knn_dst[2]
            print('dst: ', dists_dst.shape, idxs_dst.shape, pts_dst.shape)

            # translations, between src and its knn in dst
            ts_dst_src = pts_dst - points_src[:, :, None, :] 
            print('translations: ', ts_dst_src.shape)

            # define a local window for each point in src, using knn for simplicity
            knn_src = pytorch3d_ops.knn_points(points_src, points_src, lengths1=None, lengths2=None, K=9, return_nn=True, return_sorted=False)
            dists_src, idxs_src, pts_src = knn_src[0], knn_src[1], knn_src[2]
            print('src: ', dists_src.shape, idxs_src.shape, pts_src.shape)

            # all translations within in a window, between src and its knn in dst
            ts_window = pytorch3d_ops.knn_gather(ts_dst_src.view(b, l, -1), idxs_src, lengths=None)
            ts_window = ts_window.view(idxs_src.shape[0], idxs_src.shape[1], idxs_src.shape[2], 16, 3)
            print('ts_window: ', ts_window.shape)

            # quantization
            n_intervals = 3.33 // 0.1  # both directions
            intervals = torch.arange(-0.1*n_intervals, 0.1*n_intervals + 1e-8, 0.1) # [-3.3, -3.2, ..., 3.2, 3.3]
            intervals_z = torch.arange(-0.1, 0.1 + 1e-8, 0.1) # [-0.1, 0, 0.1]
            print('intervals xy: ', n_intervals, 0.1*n_intervals, intervals.shape, intervals, intervals_z)

            # tricky there, check: https://pytorch.org/docs/stable/generated/torch.bucketize.html
            x_idxs = torch.bucketize(ts_window[:, :, :, :, 0], intervals, right=True)
            y_idxs = torch.bucketize(ts_window[:, :, :, :, 1], intervals, right=True)
            z_idxs = torch.bucketize(ts_window[:, :, :, :, 2], intervals_z, right=True) 
            x_idxs -= 1
            y_idxs -= 1
            z_idxs -= 1
            print('ts_idxs: ', x_idxs.shape, y_idxs.shape, z_idxs.shape)

            # # # sanity check: find out shared translations
            # for i in range(0, x_idxs.shape[0]):
            #     for j in range(0, x_idxs.shape[1]):
            #         x_idxs_temp = x_idxs[i, j]
            #         y_idxs_temp = y_idxs[i, j]
            #         z_idxs_temp = z_idxs[i, j]
            #         for p in range(0, ts_window.shape[2]):
            #             for q in range(0, ts_window.shape[3]):
            #                 print('t, idx: ', ts_window[i, j, p, q], x_idxs[i, j, p, q], y_idxs[i, j, p, q], z_idxs[i, j, p, q])
            #         idxs_temp = torch.stack([x_idxs_temp, y_idxs_temp, z_idxs_temp], dim=2)
            #         print('idxs_temp: ', idxs_temp.shape)
            #         bin_unq, count_unq = torch.unique(idxs_temp.view(-1, 3), dim=0, return_counts=True)
            #         print('unique bins: ', bin_unq, count_unq)
        
        # process feats: gradient backpropagation during training
        # feats: knn of src in dst
        feats_dst_nn = pytorch3d_ops.knn_gather(feats_dst, idxs_dst, lengths=None)
        print('feat dst nn: ', feats_dst_nn.shape)

        # feats similarity, between src and its knn in dst
        feats = (feats_src[:, :, None, :] * feats_dst_nn).sum(-1)
        print('feats: ', feats.shape)

        # all feats within in a window, between src and its knn in dst
        feats_window = pytorch3d_ops.knn_gather(feats, idxs_src, lengths=None)
        print('feats_window: ', feats_window.shape)

        # # simple test: no feature learning 
        feats_window.fill_(1.0)

        # # # Hough voting: pytorch implemenation vs custom cuda implementation
        # 1. test pytorch implementation
        start_time_pytorch = time.time()
        b_idxs = torch.arange(0, b)[:, None, None, None].expand(b, l, 9, 16).flatten()
        p_idxs = torch.arange(0, l)[None, :, None, None].expand(b, l, 9, 16).flatten()
        x_idxs = x_idxs.flatten()
        y_idxs = y_idxs.flatten()
        z_idxs = z_idxs.flatten()

        # delete out-of-range values later
        valid = torch.all(
            torch.stack([x_idxs<len(intervals)-1, x_idxs>=0, y_idxs<len(intervals)-1, y_idxs>=0, z_idxs<len(intervals_z)-1, z_idxs>=0], dim=1),
            dim=1).flatten()
        print('valid: ', valid.shape, valid.sum())
        # build a 3d volume (x/y: -3.3 +3.3, z:-0.1 +0.1)
        vol = torch.zeros((b, l, len(intervals)-1, len(intervals)-1, len(intervals_z)-1)).float()

        # incremental voting: a 3d representation per point 
        # https://discuss.pytorch.org/t/indexing-with-repeating-indices-numpy-add-at/10223/13
        # option 1:
        # idxs = b_idxs * vol.shape[1:].numel() + p_idxs * vol.shape[2:].numel() + x_idxs * vol.shape[3:].numel() + y_idxs * vol.shape[4:].numel() + z_idxs # flatten to 1d indices
        # print('flatten to 1d indices: ', vol.shape, vol.shape[1:].numel(), vol.shape[2:].numel(), vol.shape[3:].numel(), vol.shape[4:].numel()) 
        # feats_window = feats_window.flatten()
        # print('idxs, feats_window: ', idxs.shape, feats_window.shape)
        # vol.flatten().put_(idxs[valid], feats_window[valid], accumulate=True) 
        # vol.flatten().put_(idxs[valid], torch.ones(sum(valid)).float(), accumulate=True) 
        # vol = vol.view(b, n_pts, len(intervals)-1, len(intervals)-1, len(intervals_z)-1)
        # option 2:
        vol.index_put_([b_idxs[valid], p_idxs[valid], x_idxs[valid], y_idxs[valid], z_idxs[valid]], feats_window.flatten()[valid], accumulate=True) 
        print('vote: ', vol.shape, vol.max(), vol.min(), time.time()-start_time_pytorch)

        # 2. test custom cuda implementation
        start_time_cuda = time.time()
        vol2, _, _, _ = vote(feats_window, ts_window) 
        vol2 = vol2.permute(0, 1, 3, 4, 2)
        print('vote2: ', vol2.shape, vol2.max(), vol2.min(), time.time()-start_time_cuda)

        # check result
        print('vol==vol2: ', torch.equal(vol, vol2), (vol-vol2).abs().max())

        # add conv + fc layers for final prediction
        flow = model.forward(vol.view(b*l, len(intervals)-1, len(intervals)-1, len(intervals_z)-1).permute(0, 3, 1, 2).contiguous())
        print('flow: ', flow.shape)

        # calculate loss during training
        loss = warped_pc_loss(points_src.view(-1, 3)+flow, points_dst.view(-1, 3))
        print('loss: ', loss.shape, loss)



