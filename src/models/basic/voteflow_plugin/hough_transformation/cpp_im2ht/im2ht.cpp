#include "im2ht.h"
#include "ht_cuda.h"

at::Tensor
// std::vector<at::Tensor>
im2ht_forward(
            const at::Tensor &feats,
            const at::Tensor &voxels_src,
            const at::Tensor &voxels_dst,
            const at::Tensor &idxs_src,
            const at::Tensor &idxs_dst,
            const int h,
            const int w,
            const int d
            )
{
    if (
        feats.type().is_cuda() && \
        voxels_src.type().is_cuda() && voxels_dst.type().is_cuda() && \
        idxs_src.type().is_cuda() && idxs_dst.type().is_cuda()
        )
    {
        return ht_cuda_forward(feats,
                                voxels_src, voxels_dst, 
                                idxs_src, idxs_dst, 
                                h, w, d
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor
// std::vector<at::Tensor>
im2ht_backward(const at::Tensor &grad_vol,
                const at::Tensor &voxels_src,
                const at::Tensor &voxels_dst,
                const at::Tensor &idxs_src,
                const at::Tensor &idxs_dst,
                const int h,
                const int w,
                const int d
                )
{
    if (
        grad_vol.type().is_cuda() && \
        voxels_src.type().is_cuda() && voxels_dst.type().is_cuda() && \
        idxs_src.type().is_cuda() && idxs_dst.type().is_cuda()
        )
    {
        return ht_cuda_backward(grad_vol, 
                                voxels_src, voxels_dst, 
                                idxs_src, idxs_dst, 
                                h, w, d
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("im2ht_forward", &im2ht_forward, "Forward pass of ht");
    m.def("im2ht_backward", &im2ht_backward, "Backward pass of ht");
}
