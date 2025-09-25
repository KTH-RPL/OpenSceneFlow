#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__global__ void hist_cuda_kernel(const int n,
                                              const scalar_t* X,
                                              const scalar_t* Y,
                                              const int batch, const int dim, 
                                              const int num_X, const int num_Y, 
                                              const float min_x, const float min_y, const float min_z, 
                                              const float max_x, const float max_y, const float max_z, 
                                              const int len_x, const int len_y, const int len_z,
                                              scalar_t* bins
                                              )
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    // launch in parallel:  batch * numX * numY;
    // printf("hist cuda bin size: %d, %d, %d, %d. \n", batch, len_x, len_y, len_z);
    const int b = index / num_X / num_Y % batch;
    const int i = index / num_Y % num_X;
    const int j = index  % num_Y;

    scalar_t flag_x = X[b*num_X*dim+i*dim+3];
    scalar_t flag_y = Y[b*num_Y*dim+j*dim+3];
    if (flag_x>0.0 && flag_y>0.0)
    {
      scalar_t val_x = X[b*num_X*dim+i*dim+0] - Y[b*num_Y*dim+j*dim+0];
      scalar_t val_y = X[b*num_X*dim+i*dim+1] - Y[b*num_Y*dim+j*dim+1];
      scalar_t val_z = X[b*num_X*dim+i*dim+2] - Y[b*num_Y*dim+j*dim+2];
      if (val_x >= min_x && val_x < max_x && val_y >= min_y && val_y < max_y && val_z >= min_z && val_z < max_z)
      {
        // [): left included; right excluded.
        int p_x = __float2int_rd( (val_x-min_x) / (max_x-min_x) * __int2float_rd(len_x));
        int p_y = __float2int_rd( (val_y-min_y) / (max_y-min_y) * __int2float_rd(len_y));
        int p_z = __float2int_rd( (val_z-min_z) / (max_z-min_z) * __int2float_rd(len_z));

        // printf("hist cuda coord: %d, %d, %d, %d; %d, %d, %d, %d. \n", batch, len_x, len_y, len_z, b, p_x, p_y, p_z);
        int bin_id = b*len_x*len_y*len_z + p_x*len_y*len_z  +  p_y*len_z + p_z;
        atomicAdd(bins + bin_id, 1);
      }
  }
  }
}

template <typename scalar_t>
void hist_cuda_core(cudaStream_t stream,
                              const scalar_t* X, const scalar_t* Y,
                              const int batch, const int dim, 
                              const int num_X, const int num_Y, 
                              const float min_x, const float min_y, const float min_z, 
                              const float max_x, const float max_y, const float max_z, 
                              const int len_x, const int len_y, const int len_z,
                              scalar_t* bins
                              ) 
{
  const int num_kernels = batch * num_X * num_Y;
  // printf("num kernels: %d\n", num_kernels);

  // printf("hist cuda core: %f, %f, %f; %f, %f, %f; %f, %f, %f. \n", min_x, min_y, min_z, max_x, max_y, max_z, len_x, len_y, len_z);
  // printf("hist cuda core: ", min_x, min_y, min_z, max_x, max_y, max_z, len_x, len_y, len_z, " \n");
  hist_cuda_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, 
      X, Y,
      batch, dim,
      num_X, num_Y,
      min_x, min_y, min_z, 
      max_x, max_y, max_z, 
      len_x, len_y, len_z, 
      bins
      );
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in hist_cuda_core: %s\n", cudaGetErrorString(err));
  }
}

