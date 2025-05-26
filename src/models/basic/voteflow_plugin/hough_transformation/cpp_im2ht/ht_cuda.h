#pragma once
#include <torch/extension.h>


at::Tensor
ht_cuda_forward(
                const at::Tensor &feats_src_dst,
                const at::Tensor &voxels_src,
                const at::Tensor &voxels_dst,
                const at::Tensor &idxs_src,
                const at::Tensor &idxs_dst,
                const int h,
                const int w,
                const int d
                );

// std::vector<at::Tensor>
at::Tensor
ht_cuda_backward(
                const at::Tensor &grad_output, 
                const at::Tensor &voxels_src,
                const at::Tensor &voxels_dst,
                const at::Tensor &idxs_src,
                const at::Tensor &idxs_dst,
                const int h,
                const int w,
                const int d
                );

