from torch import nn
from torch.autograd import Function
import torch
import importlib

import os, time
import hist

def histf(X, Y, min_x, min_y, min_z, max_x, max_y, max_z, len_x, len_y, len_z, mini_batch=8):
    # print('hist cuda params: ', X.shape, Y.shape,
    #       min_x, min_y, min_z,
    #       max_x, max_y, max_z,
    #       len_x, len_y, len_z,
    #       )
    histogram = hist.hist(X.contiguous(), Y.contiguous(), 
                          min_x, min_y, min_z,
                          max_x, max_y, max_z, 
                          len_x, len_y, len_z, 
                          mini_batch
                          )
    return histogram


torch.manual_seed(2022)

########################
def run_test():
    pts = torch.randn(3, 1000, 3)
    indicators = torch.randint(0, 2, size=(3, 1000, 1))
    pts1 = torch.cat([pts, indicators], dim=-1)
    pts2  = pts1.clone()
    pts2[:, :,0] += 5.
    pts2[:, :,1] += -3.
    pts2[:, :,2] += -0.2

    range_x = 10.
    range_y = 10.
    range_z = 0.5
    thres =0.1
    # bins_x = torch.linspace(-range_x, range_x, int(2*range_x/thres)+1)
    # bins_y = torch.linspace(-range_y, range_y, int(2*range_y/thres)+1)
    # bins_z = torch.linspace(-range_z, range_z, int(2*range_z/thres)+1)
    bins_x = torch.arange(-range_x, range_x+thres, thres)
    bins_y = torch.arange(-range_y, range_y+thres, thres)
    bins_z = torch.arange(-range_z, range_z+thres, thres)
    print('bins_x: ', bins_x)
    print('bins_z: ', bins_z)
    pts1 = pts1.cuda()
    pts2 = pts2.cuda()
    bins_x = bins_x.cuda()
    bins_y = bins_y.cuda()
    bins_z = bins_z.cuda()

    t_hists = histf(pts1, pts2, 
               -range_x, -range_y, -range_z,
               range_x, range_y, range_z,
               len(bins_x), len(bins_y), len(bins_z),
               )
    print('output shape: ', t_hists.shape)
    b, h, w, d = t_hists.shape
    for t_hist in t_hists:
        t_argmax = torch.argmax(t_hist)
        print(f't_argmax: {t_argmax}, {t_hist.max()} {h}, {w}, {d}, {t_argmax//d//w%h}, {t_argmax//d%w}, {t_argmax%d}')
        print('t_argmax', t_argmax//d//w%h, t_argmax//d%w, t_argmax%d, bins_x[t_argmax//d//w%h], bins_y[t_argmax//d%w], bins_z[t_argmax%d])

if __name__ == '__main__':
    
    print("Pytorch version: ", torch.__version__)
    print("GPU version: ", torch.cuda.get_device_name())
    
    run_test()