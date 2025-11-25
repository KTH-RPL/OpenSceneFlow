/*
 * Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
 * @Author: Qingwen Zhang (https://kin-zhang.github.io/)
 * @Date: 2023-08-03 16:55
 * @Description: Chamfer distance calculation between two point clouds with CUDA
 * This file is part of SeFlow (https://github.com/KTH-RPL/SeFlow).
 * If you find this repo helpful, please cite the respective publication as 
 * listed on the above website.


 * Reference: Modified from SCOOP chamfer3D [https://github.com/itailang/SCOOP]
 * faster 2x than the original version
*/

#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK 256


__global__ void NmDistanceKernel(const int pc0_n, const float *pc0_xyz, const int pc1_n, const float *pc1_xyz, float *result, int *result_i){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load point coordinates for valid threads
    float x0, y0, z0;
    int best_i = -1;
    float best = 1e20;
    
    if (tid < pc0_n) {
        x0 = pc0_xyz[tid * 3 + 0];
        y0 = pc0_xyz[tid * 3 + 1];
        z0 = pc0_xyz[tid * 3 + 2];
    }

    __shared__ float shared_pc1[THREADS_PER_BLOCK * 3];

    for (int i = 0; i < pc1_n; i += THREADS_PER_BLOCK) {
        // All threads cooperate to load shared memory
        int pc1_idx = i + threadIdx.x;
        if (pc1_idx < pc1_n) {
            shared_pc1[threadIdx.x * 3 + 0] = pc1_xyz[pc1_idx * 3 + 0];
            shared_pc1[threadIdx.x * 3 + 1] = pc1_xyz[pc1_idx * 3 + 1];
            shared_pc1[threadIdx.x * 3 + 2] = pc1_xyz[pc1_idx * 3 + 2];
        }

        __syncthreads();
 
        // Only valid threads compute distances
        if (tid < pc0_n) {
            int num_elems = min(THREADS_PER_BLOCK, pc1_n - i);
            for (int j = 0; j < num_elems; j++) {
                float x1 = shared_pc1[j * 3 + 0];
                float y1 = shared_pc1[j * 3 + 1];
                float z1 = shared_pc1[j * 3 + 2];
                float d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + (z1 - z0) * (z1 - z0);
                if (d < best) {
                    best = d;
                    best_i = j + i;
                }
            }
        }

        __syncthreads();
    }
    
    // Only valid threads write results
    if (tid < pc0_n) {
        result[tid] = best;
        result_i[tid] = best_i;
    }
}

int chamfer_cuda_forward(const at::Tensor &pc0, const at::Tensor &pc1, at::Tensor &dist0, at::Tensor &dist1, at::Tensor &idx0, at::Tensor &idx1)
{
	at::cuda::CUDAGuard device_guard(pc0.device());
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	const int pc0_n = pc0.size(0);
	const int pc1_n = pc1.size(0);

	const int col_blocks_pc0 = DIVUP(pc0_n, THREADS_PER_BLOCK);
	dim3 blocks_pc0(col_blocks_pc0);
	const int col_blocks_pc1 = DIVUP(pc1_n, THREADS_PER_BLOCK);
	dim3 blocks_pc1(col_blocks_pc1);
	dim3 threads(THREADS_PER_BLOCK);
	
	NmDistanceKernel<<<blocks_pc0, threads, 0, stream>>>(pc0_n, pc0.data_ptr<float>(), pc1_n, pc1.data_ptr<float>(), dist0.data_ptr<float>(), idx0.data_ptr<int>());
	NmDistanceKernel<<<blocks_pc1, threads, 0, stream>>>(pc1_n, pc1.data_ptr<float>(), pc0_n, pc0.data_ptr<float>(), dist1.data_ptr<float>(), idx1.data_ptr<int>());
	
	AT_CUDA_CHECK(cudaGetLastError());

	return 1;
}

__global__ void NmDistanceGradKernel(const int pc0_n, const float *pc0_xyz, const int pc1_n, const float *pc1_xyz, 
                                     const float *grad_dist0, const int *idx0, float *grad_pc0, float *grad_pc1)
{
	CUDA_1D_KERNEL_LOOP(j0, pc0_n){
        float x0 = pc0_xyz[j0 * 3 + 0];
        float y0 = pc0_xyz[j0 * 3 + 1];
        float z0 = pc0_xyz[j0 * 3 + 2];

        int j1 = idx0[j0];
        float x1 = pc1_xyz[j1 * 3 + 0];
        float y1 = pc1_xyz[j1 * 3 + 1];
        float z1 = pc1_xyz[j1 * 3 + 2];

        float g = grad_dist0[j0] * 2;

        atomicAdd(&grad_pc0[j0 * 3 + 0], g * (x0 - x1));
        atomicAdd(&grad_pc0[j0 * 3 + 1], g * (y0 - y1));
        atomicAdd(&grad_pc0[j0 * 3 + 2], g * (z0 - z1));
        
        atomicAdd(&grad_pc1[j1 * 3 + 0], - (g * (x0 - x1)));
        atomicAdd(&grad_pc1[j1 * 3 + 1], - (g * (y0 - y1)));
        atomicAdd(&grad_pc1[j1 * 3 + 2], - (g * (z0 - z1)));
    }
}
int chamfer_cuda_backward(const at::Tensor &pc0, const at::Tensor &pc1, 
                          const at::Tensor &idx0, const at::Tensor &idx1,
                          at::Tensor &grad_dist0, at::Tensor &grad_dist1, 
                          at::Tensor &grad_pc0, at::Tensor &grad_pc1)
{
	at::cuda::CUDAGuard device_guard(pc0.device());
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	const int pc0_n = pc0.size(0);
	const int pc1_n = pc1.size(0);

	const int col_blocks_pc0 = DIVUP(pc0_n, THREADS_PER_BLOCK);
	dim3 blocks_pc0(col_blocks_pc0);
	const int col_blocks_pc1 = DIVUP(pc1_n, THREADS_PER_BLOCK);
	dim3 blocks_pc1(col_blocks_pc1);
	dim3 threads(THREADS_PER_BLOCK);
	
	NmDistanceGradKernel<<<blocks_pc0, threads, 0, stream>>>(pc0_n, pc0.data_ptr<float>(), pc1_n, pc1.data_ptr<float>(), grad_dist0.data_ptr<float>(), idx0.data_ptr<int>(), grad_pc0.data_ptr<float>(), grad_pc1.data_ptr<float>());
	NmDistanceGradKernel<<<blocks_pc1, threads, 0, stream>>>(pc1_n, pc1.data_ptr<float>(), pc0_n, pc0.data_ptr<float>(), grad_dist1.data_ptr<float>(), idx1.data_ptr<int>(), grad_pc1.data_ptr<float>(), grad_pc0.data_ptr<float>());
	
	AT_CUDA_CHECK(cudaGetLastError());

	return 1;
}