#pragma once
#include <torch/extension.h>

at::Tensor
hist_cuda(const at::Tensor &X, const at::Tensor &Y,
        const float min_x, const float min_y, const float min_z,
        const float max_x, const float max_y, const float max_z,
        const int len_x, const int len_y, const int len_z,
        const int mini_batch
        );
