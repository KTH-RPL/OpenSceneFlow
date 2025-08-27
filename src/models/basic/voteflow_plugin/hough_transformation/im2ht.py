import os
import warnings
from glob import glob

from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, "cpp_im2ht")
    tar_dir = os.path.join(src_dir, "build", ext_name)
    os.makedirs(tar_dir, exist_ok=True)
    srcs = glob(f"{src_dir}/*.cu") + glob(f"{src_dir}/*.cpp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.utils.cpp_extension import load
        ext = load(
            name=ext_name,
            sources=srcs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[],
            build_directory=tar_dir,
        )
    return ext

# defer calling load_cpp_ext to make CUDA_VISIBLE_DEVICES happy
im2ht = None


class IM2HTFunction(Function):
    @staticmethod
    def forward(ctx,
                feats_src_dst,
                voxels_src,
                voxels_dst,
                idxs_src,
                idxs_dst,
                h, w, d
                ):

        ctx.h = h
        ctx.w = w
        ctx.d = d
        ctx.save_for_backward(voxels_src, voxels_dst, idxs_src, idxs_dst)
        vol = im2ht.im2ht_forward(
            feats_src_dst,
            voxels_src,
            voxels_dst,
            idxs_src,
            idxs_dst,
            h, 
            w, 
            d
        )
        return vol

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_vol):
        voxels_src, voxels_dst, idxs_src, idxs_dst = ctx.saved_tensors  # it is a list of saved tensors!

        grad_input_src_dst = im2ht.im2ht_backward(
            grad_vol,
            voxels_src,
            voxels_dst,
            idxs_src,
            idxs_dst,
            ctx.h,
            ctx.w,
            ctx.d
        )

        return (
            grad_input_src_dst, 
            None, # voxels_src,
            None, # voxels_dst,
            None, # idxs_src,
            None, # idxs_dst,
            None, # h
            None, # w
            None  # d
        )


class IM2HT(nn.Module):
    def __init__(self, ):
        super(IM2HT, self).__init__()
        
        global im2ht
        im2ht = load_cpp_ext("im2ht")

    def forward(self, feats_src_dst, voxels_src, voxels_dst, idxs_src, idxs_dst, h, w, d):  
        # print('IM2HT forward', feats.shape, idxs_fps.shape, idxs_src.shape, idxs_dst.shape, bins_x.shape, bins_y.shape, bins_z.shape)
        return IM2HTFunction.apply(
            feats_src_dst.contiguous(),
            voxels_src.contiguous(),
            voxels_dst.contiguous(),
            idxs_src.contiguous(),
            idxs_dst.contiguous(),
            h, w, d
        )
