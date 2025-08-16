#include <vector>
#include "hist_cuda_core.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

// extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu


at::Tensor
hist_cuda(const at::Tensor &X, const at::Tensor &Y,
        const float min_x, const float min_y, const float min_z,
        const float max_x, const float max_y, const float max_z,
        const int len_x, const int len_y, const int len_z,
        const int mini_batch
            )
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));

    AT_ASSERTM(X.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(Y.is_contiguous(), "input tensor has to be contiguous");

    AT_ASSERTM(X.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(Y.type().is_cuda(), "input must be a CUDA tensor");

    const int batch = X.size(0); 
    const int num_X = X.size(1);
    const int dim = X.size(2);
    const int num_Y = Y.size(1);

    AT_ASSERTM((X.size(0) == Y.size(0)), "batch_X (%d) != batch_Y (%d).", X.size(0), Y.size(0));
    AT_ASSERTM((X.size(2) == Y.size(2)), "dim_X (%d) != dim_Y (%d).", X.size(2), Y.size(2));

    AT_ASSERTM((dim == 4), "dim (%d) != 4; 3 for (x, y, z); 1 for indicator,padded or not.", dim);

    // printf("len: %d %d %f \n", len_x, len_y, len_z);
    // printf("hist cuda coord: %f, %f, %f; %f, %f, %f; %f, %f, %f. \n", val_x, val_y, val_z, p_x, p_y, p_z, len_x, len_y, len_z);

    // auto bins = at::zeros({batch, len_x, len_y, len_z}, X.options());
    // AT_DISPATCH_FLOATING_TYPES(X.type(), "hist_cuda_core", ([&] {
    //     hist_cuda_core(at::cuda::getCurrentCUDAStream(),
    //                                     X.data<scalar_t>(), Y.data<scalar_t>(),
    //                                     batch, dim, num_X, num_Y,
    //                                     min_x, min_y, min_z, 
    //                                     max_x, max_y, max_z, 
    //                                     len_x, len_y, len_z, 
    //                                     bins.data<scalar_t>());
    //     }));

    auto bins = at::zeros({batch, len_x, len_y, len_z}, X.options());

    int iters = batch / mini_batch; 
    if (batch % mini_batch != 0) 
    { 
        iters += 1; 
    }

    for (int i=0; i<iters; ++i)
    {
        int mini_batch_ = mini_batch;
        if ((i+1) * mini_batch > batch) 
        {
            mini_batch_ = batch - i * mini_batch; 
        }
        // printf("iter: %d %d %d %d %d \n", i, iters, mini_batch_, mini_batch, batch);
        AT_DISPATCH_FLOATING_TYPES(X.type(), "hist_cuda_core", ([&] {
            hist_cuda_core(at::cuda::getCurrentCUDAStream(),
                                            X.data<scalar_t>() + i*mini_batch*num_X*dim, 
                                            Y.data<scalar_t>() + i*mini_batch*num_Y*dim, 
                                            mini_batch_, dim, num_X, num_Y,
                                            min_x, min_y, min_z, 
                                            max_x, max_y, max_z, 
                                            len_x, len_y, len_z, 
                                            bins.data<scalar_t>()+i*mini_batch*len_x*len_y*len_z);
            }));
    }



    return bins;
}
