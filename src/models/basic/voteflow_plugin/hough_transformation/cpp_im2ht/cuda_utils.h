// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
//#if __CUDA_ARCH__ >= 200
//    const int CUDA_NUM_THREADS = 1024;
//#else
//    const int CUDA_NUM_THREADS = 512;
//#endif
const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


// #ifndef CUDA_NUM_THREADS 
// #define CUDA_NUM_THREADS 1024
// #endif

// #define CUDA_KERNEL_LOOP(i, n)                          \
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
//       i < (n);                                          \
//       i += blockDim.x * gridDim.x)

// const int CUDA_NUM_THREADS = 1024;

// inline int GET_BLOCKS(const int N)
// {
//   return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
// }


// // C++ interface
// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// // #define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// #define CHECK_CUDA(x)                                          \
//   do {                                                         \
//     AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
//   } while (0)

// #define CHECK_CONTIGUOUS(x)                                         \
//   do {                                                              \
//     AT_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
//   } while (0)

// #define CHECK_IS_INT(x)                              \
//   do {                                               \
//     AT_CHECK(x.scalar_type() == at::ScalarType::Int, \
//              #x " must be an int tensor");           \
//   } while (0)

// #define CHECK_IS_FLOAT(x)                              \
//   do {                                                 \
//     AT_CHECK(x.scalar_type() == at::ScalarType::Float, \
//              #x " must be a float tensor");            \
//   } while (0)
