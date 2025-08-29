"""
# Created: 2023-08-04 11:20
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of SeFlow (https://github.com/KTH-RPL/SeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# 
# Description: ChamferDis speedup using CUDA
"""
from torch import nn
from torch.autograd import Function
import torch

import os, time
import chamfer3D
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))


# GPU tensors only
class ChamferDis(Function):
    @staticmethod
    def forward(ctx, pc0, pc1):
        # pc0: (N,3), pc1: (M,3)
        dis0 = torch.zeros(pc0.shape[0]).to(pc0.device).contiguous()
        dis1 = torch.zeros(pc1.shape[0]).to(pc1.device).contiguous()
        
        idx0 = torch.zeros(pc0.shape[0], dtype=torch.int32).to(pc0.device).contiguous()
        idx1 = torch.zeros(pc1.shape[0], dtype=torch.int32).to(pc1.device).contiguous()


        chamfer3D.forward(pc0, pc1, dis0, dis1, idx0, idx1)
        ctx.save_for_backward(pc0, pc1, idx0, idx1)
        return dis0, dis1, idx0, idx1

    @staticmethod
    def backward(ctx, grad_dist0, grad_dist1, grad_idx0, grad_idx1):
        pc0, pc1, idx0, idx1 = ctx.saved_tensors
        grad_dist0 = grad_dist0.contiguous()
        grad_dist1 = grad_dist1.contiguous()
        device = grad_dist1.device

        grad_pc0 = torch.zeros(pc0.size()).to(device).contiguous()
        grad_pc1 = torch.zeros(pc1.size()).to(device).contiguous()

        chamfer3D.backward(
            pc0, pc1, idx0, idx1, grad_dist0, grad_dist1, grad_pc0, grad_pc1
        )
        return grad_pc0, grad_pc1
    
class nnChamferDis(nn.Module):
    def __init__(self, truncate_dist=True):
        super(nnChamferDis, self).__init__()
        self.truncate_dist = truncate_dist

    def forward(self, input0, input1, truncate_dist=-1):
        input0 = input0.contiguous()
        input1 = input1.contiguous()
        dist0, dist1, _, _ = ChamferDis.apply(input0, input1)

        if truncate_dist<=0:
            return torch.mean(dist0) + torch.mean(dist1)

        valid_mask0 = (dist0 <= truncate_dist)
        valid_mask1 = (dist1 <= truncate_dist)
        truncated_sum = torch.nanmean(dist0[valid_mask0]) + torch.nanmean(dist1[valid_mask1])
        return truncated_sum

    def dis_res(self, input0, input1):
        input0 = input0.contiguous()
        input1 = input1.contiguous()
        dist0, dist1, _, _ = ChamferDis.apply(input0, input1)
        return dist0, dist1
    
    def truncated_dis(self, input0, input1, truncate_dist=2):
        # nsfp: truncated distance way is set >= 2 to 0 but not nanmean
        cham_x, cham_y = self.dis_res(input0, input1)
        cham_x[cham_x >= truncate_dist] = 0.0
        cham_y[cham_y >= truncate_dist] = 0.0
        return torch.mean(cham_x) + torch.mean(cham_y)
    
    def disid_res(self, input0, input1):
        input0 = input0.contiguous()
        input1 = input1.contiguous()
        dist0, dist1, idx0, idx1 = ChamferDis.apply(input0, input1)
        return dist0, dist1, idx0, idx1
class NearestNeighborDis(nn.Module):
    def __init__(self):
        super(NearestNeighborDis, self).__init__()

    def forward(self, input0, input1):
        input0 = input0.contiguous()
        input1 = input1.contiguous()
        dist0, dist1, _, _ = ChamferDis.apply(input0, input1)

        return torch.mean(dist0[dist0 <= 2])
    
if __name__ == "__main__":
    import numpy as np
    pc0 = np.load(f'{BASE_DIR}/assets/tests/test_pc0.npy')
    pc1 = np.load(f'{BASE_DIR}/assets/tests/test_pc1.npy')
    print('0: {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))
    pc0 = torch.from_numpy(pc0[...,:3]).float().cuda().contiguous()
    pc1 = torch.from_numpy(pc1[...,:3]).float().cuda().contiguous()
    pc0.requires_grad = True
    pc1.requires_grad = True
    print(pc0.shape, "demo data: ", pc0[0])
    print(pc1.shape, "demo data: ", pc1[0])
    print('1: {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))

    start_time = time.time()
    loss = nnChamferDis(truncate_dist=False)(pc0, pc1)
    loss.backward()
    print("loss: ", loss)
    print(f"Chamfer Distance Cal time: {(time.time() - start_time)*1000:.3f} ms")