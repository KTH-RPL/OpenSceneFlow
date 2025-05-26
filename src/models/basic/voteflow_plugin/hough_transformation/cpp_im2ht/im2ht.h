#pragma once
#include <torch/extension.h>

// std::vector<at::Tensor>
at::Tensor
im2ht_forward(
            const at::Tensor &feats_src_dst,
            const at::Tensor &voxels_src,
            const at::Tensor &voxels_dst,
            const at::Tensor &idxs_src,
            const at::Tensor &idxs_dst,
            const int h,
            const int w,
            const int d
            );

at::Tensor
// std::vector<at::Tensor>
im2ht_backward(
            const at::Tensor &grad_output,
            const at::Tensor &voxels_src,
            const at::Tensor &voxels_dst,
            const at::Tensor &idxs_src,
            const at::Tensor &idxs_dst,
            const int h,
            const int w,
            const int d
            );
