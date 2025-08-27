import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

import random
import numpy as np
import math
import scipy.io as sio
import glob

from models.ht.ht_cuda import HT_CUDA
from models.ht.im2ht import IM2HTFunction

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


if torch.cuda.is_available():
    # device_name = "cuda"
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed(0)

    device_name = "cuda"
    torch.cuda.manual_seed(0)
    # torch.backends.cudnn.enabled = False  #greatly slow down the speed
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(0)
    torch.cuda.empty_cache()
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")
else:
    device_name = "cpu"
    print("CUDA is not available")
device = torch.device(device_name)
print('device: ', device)


b = 2
l = 64
c=3
m = 4
n = 8
max_x=0.5
max_y=0.5
max_z=0.2
voxel=0.1
nx = math.ceil(max_x / voxel) * 2  # assume 120km/h, along both directions
ny = math.ceil(max_y / voxel) * 2  # assume 120km/h, along both directions
nz = math.ceil(max_z / voxel) * 2  

vote = HT_CUDA(nx, ny, nz)
vote = vote.double().to(device)

print('grad check***********')    

feats = torch.rand(size=(b, l, n, 3), requires_grad=True).double().to(device)
voxels_src = torch.randint(size=(b, l, 2), low=0, high=nx).double().to(device)
voxels_dst = torch.randint(size=(b, l, 2), low=0, high=nx).double().to(device)
idxs_src = torch.randint(0, l, size=(b, l, m)).double().to(device)
idxs_dst = torch.randint(0, l, size=(b, l, n)).double().to(device)
print('input shapes: ', feats.shape, voxels_src.shape, voxels_dst.shape, idxs_src.shape, idxs_dst.shape)
# grad_output = torch.randn(vol.shape).double().to(device)
# vol.backward(gradient=grad_output)
# grad_input = fs_window.grad
# print(grad_input.shape)

# only able to test inputs with tiny sizes
res = gradcheck(vote, (feats, voxels_src, voxels_dst, idxs_src, idxs_dst), raise_exception=True)
# res = gradcheck(IM2HTFunction.apply, (feats, bins_x, bins_y, bins_z, flags, idxs, m, n, h, w, d), raise_exception=True)
# # res=gradcheck(myconv, input, eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True)
print('grad check', res)
