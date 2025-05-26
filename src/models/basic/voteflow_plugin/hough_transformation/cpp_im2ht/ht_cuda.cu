#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "im2ht_cuda.cuh"


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
        )
{
    AT_ASSERTM(feats_src_dst.is_contiguous(), "feats tensor has to be contiguous");
    AT_ASSERTM(voxels_src.is_contiguous(), "voxels tensor has to be contiguous");
    AT_ASSERTM(voxels_dst.is_contiguous(), "voxels tensor has to be contiguous");
    AT_ASSERTM(idxs_src.is_contiguous(), "idxs tensor has to be contiguous");
    AT_ASSERTM(idxs_dst.is_contiguous(), "idxs tensor has to be contiguous");

    // AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");

    const int b = feats_src_dst.size(0);
    const int l = feats_src_dst.size(1); // num of points
    const int c = feats_src_dst.size(-1); // num of channels
    const int m = idxs_src.size(2); // m
    const int n = idxs_dst.size(2); // n

    // AT_ASSERTM(m_ == m && n_ <= n && k_ == k,
    //     "input shape and predefined shape do not match: (%04d x %04d x %04d vs %04d x %04d x %04d).", k_, m_, n_, k, m, n);
 
    // printf("dimensions b=%06d, l=%06d, m=%06d, n=%06d, h=%06d, w= %06d, d= %06d \n",
    //                       b, l, m, n, h, w, d);
    auto vol_ht = at::zeros({b, l, c, h, w}, feats_src_dst.options());

    AT_DISPATCH_FLOATING_TYPES(vol_ht.type(), "im2ht_cuda_forward", ([&] {
        im2ht_cuda_forward(at::cuda::getCurrentCUDAStream(),
                    feats_src_dst.data<scalar_t>() ,
                    vol_ht.data<scalar_t>(),
                    voxels_src.data<scalar_t>() ,
                    voxels_dst.data<scalar_t>(),
                    idxs_src.data<scalar_t>(),
                    idxs_dst.data<scalar_t>(),
                    b, l, c,
                    m, n, 
                    h, w, d
                    );

    }));
    // std::cout <<"output" <<output.sum() << std::endl;
    // printf("output", output.sum());
    return vol_ht;
}



// std::vector<at::Tensor>
at::Tensor
ht_cuda_backward(
            const at::Tensor &grad_vol, 
            const at::Tensor &voxels_src,
            const at::Tensor &voxels_dst,
            const at::Tensor &idxs_src,
            const at::Tensor &idxs_dst,
            const int h,
            const int w,
            const int d
            )
{

    AT_ASSERTM(grad_vol.is_contiguous(), "grad_output tensor has to be contiguous");
    AT_ASSERTM(voxels_src.is_contiguous(), "voxels tensor has to be contiguous");
    AT_ASSERTM(voxels_dst.is_contiguous(), "voxels tensor has to be contiguous");
    AT_ASSERTM(idxs_src.is_contiguous(), "idxs tensor has to be contiguous");
    AT_ASSERTM(idxs_dst.is_contiguous(), "idxs tensor has to be contiguous");

    // AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");

    // grad_output: [b, l, c, h, w]
    const int b = grad_vol.size(0);
    const int l = grad_vol.size(1);
    const int c = grad_vol.size(2);

    const int m = idxs_src.size(2); // m
    const int n = idxs_dst.size(2); // n

    // printf("h_ = %04d, w_ = %04d, d_ = %04d vs h = %04d, w = %04d, d = %04d\n", h_, w_, d_, h, w, d);
    // AT_ASSERTM(h_ == h && w_ == w && d_== d,
    //     "grad_out shape and predefined shape do not match: ", h_, ' ', w_, ' ', d_, 'vs', h, ' ', w, ' ', d);

    auto grad_feats_src_dst = at::zeros({b, l, n, c}, grad_vol.options());
    
    AT_DISPATCH_FLOATING_TYPES(grad_vol.type(), "im2ht_cuda_backward", ([&] {
        im2ht_cuda_backward(at::cuda::getCurrentCUDAStream(),
                    grad_feats_src_dst.data<scalar_t>(),
                    grad_vol.data<scalar_t>(),
                    voxels_src.data<scalar_t>(),
                    voxels_dst.data<scalar_t>(),
                    idxs_src.data<scalar_t>(),
                    idxs_dst.data<scalar_t>(),
                    b, l, c,
                    m, n, 
                    h, w, d
                );

    }));

    return grad_feats_src_dst; 
}
