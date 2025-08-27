#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#include "cuda_utils.h"

template <typename scalar_t>
__global__ void im2ht_cuda_forward_kernel(const int64_t n_threads,
                                  const scalar_t *feats, 
                                  scalar_t *vol_ht,
                                  const scalar_t *voxels_src,
                                  const scalar_t *voxels_dst,
                                  const scalar_t *idxs_src,
                                  const scalar_t *idxs_dst,
                                  const int b, const int l, const int c,
                                  const int m, const int n,
                                  const int h, const int w, const int d
                                )
{
  CUDA_KERNEL_LOOP(index, n_threads)
  {
    // [b, l, c, m, n]
    int b_tmp = index / l / c / m / n % b;  // b
    int l_tmp = index / c / m / n % l;  // l
    int c_tmp = index / m / n % c;  // c
    int m_tmp = index / n % m; // m
    int n_tmp = index % n; // n

    int src_tmp = idxs_src[b_tmp * (l * m) + l_tmp * m + m_tmp];  // point idx in [b, l, m]
    if (src_tmp>=0 )
    {
      int dst_tmp = idxs_dst[b_tmp * (l * n) + src_tmp * n + n_tmp];  // point idx in [b, l, n]
      if (dst_tmp>=0)
      {
        // Z-Y-X after voxelization
        int y_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 0];  // in [b, l, 2]
        int x_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 1];  // in [b, l, 2]

        int y_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 0];  // in [b, l, 2]
        int x_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 1];  // in [b, l, 2]

        scalar_t feat_tmp = feats[b_tmp * (l * n * c) + src_tmp * (n * c) + n_tmp * (c) + c_tmp];  // in [b, l, n, c]

        int bin_x = x_dst_tmp - x_src_tmp;
        int bin_y = y_dst_tmp - y_src_tmp;
        bin_x += w/2;
        bin_y += h/2;

        if (y_src_tmp >=0 && x_src_tmp >=0 && y_dst_tmp >= 0 && x_dst_tmp >=0)
        {
          if (bin_x >=0 && bin_x < w && bin_y >= 0 && bin_y < h)
          {
            // (0, 0) is at position (h//2, w//2)
            // vol_ht: [b, l, h, w]; 
            int64_t offset_vol_tmp =  b_tmp * (l * c * h * w) + l_tmp * (c * h * w) + c_tmp * (h * w) + bin_y * w + bin_x;
            atomicAdd(vol_ht+offset_vol_tmp, feat_tmp);
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void im2ht_cuda_backward_kernel(const int n_threads,
                                    scalar_t* grad_feats,
                                    const scalar_t* grad_vol_ht, 
                                    const scalar_t *voxels_src,
                                    const scalar_t *voxels_dst,
                                    const scalar_t *idxs_src,
                                    const scalar_t *idxs_dst,
                                    const int b, const int l, const int c, 
                                    const int m, const int n,
                                    const int h, const int w, const int d
                                  )
{
  // todo: coalesce
  CUDA_KERNEL_LOOP(index, n_threads)
  {
    // [b, l, c, m, n]
    int b_tmp = index / l / c / m / n % b; // b
    int l_tmp = index / c / m / n % l;  // l
    int c_tmp = index / m / n % c;  // c
    int m_tmp = index / n % m; // m
    int n_tmp = index % n; // n

    int src_tmp = idxs_src[b_tmp * (l * m) + l_tmp * m + m_tmp];  // point idx in [b, l, m]
    if (src_tmp>=0 )
    {
      int dst_tmp = idxs_dst[b_tmp * (l * n) + src_tmp * n + n_tmp];  // point idx in [b, l, n]
      if (dst_tmp>=0)
      {
        // Z-Y-X after voxelization
        int y_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 0];  // in [b, l, 2]
        int x_src_tmp = voxels_src[b_tmp * (l * 2) + src_tmp * 2 + 1];  // in [b, l, 2]

        int y_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 0];  // in [b, l, 2]
        int x_dst_tmp = voxels_dst[b_tmp * (l * 2) + dst_tmp * 2 + 1];  // in [b, l, 2]

        int bin_x = x_dst_tmp - x_src_tmp;
        int bin_y = y_dst_tmp - y_src_tmp;
        bin_x += w/2;
        bin_y += h/2;

        if (y_src_tmp >=0 && x_src_tmp >=0 && y_dst_tmp >= 0 && x_dst_tmp >=0)
        {
          if (bin_x >=0 && bin_x < w && bin_y >= 0 && bin_y < h)
          {
            // (0, 0) is at position (h//2, w//2)
            //  vol_ht: [b, l, c, h, w];
            // output gradients
            int offset_vol_tmp =  b_tmp * (l * c * h * w) + l_tmp * (c * h * w) + c_tmp * (h * w) + bin_y * w + bin_x; // [b, l, c, h, w]
            scalar_t grad_vol_tmp = grad_vol_ht[offset_vol_tmp];  // 
            // input grads
            int offset_feat_tmp = b_tmp * (l * n * c) + src_tmp * (n * c) + n_tmp * (c) + c_tmp;  // in [b, l, n, c]
            atomicAdd(grad_feats+offset_feat_tmp, grad_vol_tmp);
          }
        }
      }
    }
  }
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% //
template <typename scalar_t>
void im2ht_cuda_forward(cudaStream_t stream,
                  const scalar_t* feats, 
                  scalar_t* vol_ht,
                  const scalar_t* voxels_src,
                  const scalar_t* voxels_dst,
                  const scalar_t* idxs_src,
                  const scalar_t* idxs_dst,
                  const int b, const int l, const int c,
                  const int m, const int n, 
                  const int h, const int w, const int d
                ) 
{
  const int num_kernels = b * l * c * m * n;
  // printf("dimensions num_kernels=%16d, b=%06d, l=%06d, m=%06d, n=%06d, h=%06d, w= %06d, d= %06d \n",
  //                         num_kernels, b, l, m, n, h, w, d);
  im2ht_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          feats, 
                                                          vol_ht,
                                                          voxels_src,
                                                          voxels_dst,
                                                          idxs_src,
                                                          idxs_dst,
                                                          b, l, c,
                                                          m, n, 
                                                          h, w, d
                                                        );
                                                      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in im2ht_gpu_kernel: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void im2ht_cuda_backward(cudaStream_t stream,
                  scalar_t* grad_feats,
                  const scalar_t* grad_vol, 
                  const scalar_t* voxels_src,
                  const scalar_t* voxels_dst,
                  const scalar_t* idxs_src,
                  const scalar_t* idxs_dst,
                  const int b, const int l, const int c, 
                  const int m, const int n, 
                  const int h, const int w, const int d
                  )
{

  const int num_kernels = b * l * c * m * n;
  // ***********************************************************************//
  im2ht_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_feats, 
                                                          grad_vol, 
                                                          voxels_src,
                                                          voxels_dst,
                                                          idxs_src,
                                                          idxs_dst,
                                                          b, l, c,
                                                          m, n, 
                                                          h, w, d
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ht2im_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

