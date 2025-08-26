import torch.nn as nn
from .im2ht import IM2HT

class HT_CUDA(nn.Module):
    """
    for each voxel in src:
        find its m knn voxels in src:
            for each voxel_src:
                find its n knn voxels in dst; 
                    calculate a transaltion which determines the bin in voting (bin_y, bin_x); 
                    vote incrementally by the amount of correponding src and dst features;
    """
    """ mapping from points to histgrams
        input: feats_src_dst [b, l, n, c]
        input: voxels_src [b, l, 2]
        input: voxels_dst [b, l, 2]
        input: idxs_src [b, l, m]
        input: idxs_dst [b, l, n]
        input: bins_x int 
        input: bins_y int 
        input: bins_z int
        output: HT feaures [b, l, h, w], c*2 stands for the concatenation of features from src and dst.
    """
    def __init__(self, h, w, d):
        super(HT_CUDA, self).__init__()
        self.h, self.w, self.d = h, w, d
        assert h%2==0
        assert w%2==0
        self.ht = IM2HT()

        self.__repr__()

    def __repr__(self):
        return self.__class__.__name__ + f'( h: {self.h:04d}, w: {self.w:04d}, d: {self.d:04d} )'

    def forward(self, feats_src_dst, voxels_src, voxels_dst, idxs_src, idxs_dst):
        # print('ht_cuda forward: ', feats.shape, idxs_fps.shape, idxs_src.shape, idxs_dst.shape, bins_x.shape, bins_y.shape, bins_z.shape)
        # print(f'( k:{self.k:04d}, m: {self.m:04d}, n: {self.n:04d}, h: {self.h:04d}, w: {self.w:04d}, d: {self.d:04d} )')
        assert feats_src_dst.dim()==4
        assert voxels_src.shape == voxels_dst.shape

        b, l, n = idxs_dst.shape
        _, _, m = idxs_src.shape

        datatype = feats_src_dst.dtype
        voxels_src = voxels_src.to(datatype)
        voxels_dst = voxels_dst.to(datatype)
        idxs_src = idxs_src.to(datatype)
        idxs_dst = idxs_dst.to(datatype)
        vol = self.ht(feats_src_dst, voxels_src, voxels_dst, idxs_src, idxs_dst, self.h, self.w, self.d)
        # print('vol: ', vol.shape, vol.min(), vol.max())
        return vol
