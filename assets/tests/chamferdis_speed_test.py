"""
# Created: 2023-08-09 23:40
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
#
# Description: Test which existing chamfer distance is faster.

Dependence for this test scripts:
    * faiss-gpu
    * Pytorch3d
    * mmcv
"""
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
BASEF_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..'))
sys.path.append(BASE_DIR)
sys.path.append(BASEF_DIR)

import torch
import numpy as np
import time

FAISS_TEST = True
PYTORCH3D_TEST = True
MMCV_TEST = False
CUDA_TEST = True

if __name__ == "__main__":
    pc0 = np.load(f'{BASEF_DIR}/assets/tests/test_pc0.npy')
    pc1 = np.load(f'{BASEF_DIR}/assets/tests/test_pc1.npy')
    print('Start status on GPU allocation: {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))
    pc0 = torch.from_numpy(pc0[...,:3]).float().cuda().contiguous()
    pc1 = torch.from_numpy(pc1[...,:3]).float().cuda().contiguous()
    pc0.requires_grad = True
    pc1.requires_grad = True

    print(pc0.shape, "demo data: ", pc0[0])
    print(pc1.shape, "demo data: ", pc1[0])
    print('Status after loading data: {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))

    time.sleep(1)
    if FAISS_TEST:
        print("------ START Faiss Chamfer Distance Cal ------")
        import faiss
        import faiss.contrib.torch_utils
        def faiss_chamfer_distance(pc1, pc2):
            def faiss_knn(pc1, pc2):
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, 3)
                index.add(pc2)
                distances, indices = index.search(pc1, 1) # [N_1, 1]
                return distances
            pc1_knn = faiss_knn(pc1, pc2)
            pc2_knn = faiss_knn(pc2, pc1)

            # # NOTE: truncated Chamfer distance.
            # dist_thd = 2
            # pc1_knn[pc1_knn >= dist_thd] = 0.0
            # pc2_knn[pc2_knn >= dist_thd] = 0.0

            # NOTE: Chamfer distance. mean based on pts number
            cham_pc1 = pc1_knn.mean()
            cham_pc2 = pc2_knn.mean()

            return cham_pc1 + cham_pc2
        
        start_time = time.time()
        loss = faiss_chamfer_distance(pc0, pc1)
        print("loss: ", loss)
        print(f"Faiss Chamfer Distance Cal time: {(time.time() - start_time)*1000:.3f} ms]\n")

    if PYTORCH3D_TEST:
        print("------ START Pytorch3d Chamfer Distance Cal ------")
        from src.models.basic.nsfp_module import my_chamfer_fn
        start_time = time.time()
        loss0, _ = my_chamfer_fn(pc0.unsqueeze(0), pc1.unsqueeze(0), truncate_dist=False)

        print("loss: ", loss0)
        print(f"Pytorch3d Chamfer Distance Cal time: {(time.time() - start_time)*1000:.3f} ms")
        print()

    if MMCV_TEST:
        # NOTE: mmcv chamfer distance is on x,y only.. That's why the result is not correct.
        # DONE: the result is not correct.... 
        print("------ START mmcv 2D Chamfer Distance Cal ------")
        from mmcv.ops import chamfer_distance
        start_time = time.time()
        dis0, dis1, idx0, idx1= chamfer_distance(pc0.unsqueeze(0), pc1.unsqueeze(0))
        loss_0t1 = torch.sum(dis0) / pc0.shape[0]
        loss_1t0 = torch.sum(dis1) / pc1.shape[0]

        loss = loss_0t1 + loss_1t0
        print("loss: ", loss)
        print(f"mmcv Chamfer Distance Cal time: {(time.time() - start_time)*1000:.3f} ms\n")

    if CUDA_TEST:
        print("------ START CUDA Chamfer Distance Cal ------")
        from assets.cuda.chamfer3D import nnChamferDis
        start_time = time.time()
        loss = nnChamferDis(truncate_dist=False)(pc0, pc1)
        print("loss: ", loss)
        print(f"Chamfer Distance Cal time: {(time.time() - start_time)*1000:.3f} ms")
        print()

"""
0: 0.000MB
torch.Size([88132, 3]) demo data:  tensor([-8.2266,  8.3516,  1.4922], device='cuda:0', grad_fn=<SelectBackward0>)
torch.Size([88101, 3]) demo data:  tensor([-7.9961,  8.1328,  0.4839], device='cuda:0', grad_fn=<SelectBackward0>)
1: 2.017MB

------ START Faiss Chamfer Distance Cal ------
loss:  tensor(0.1710, device='cuda:0')
Faiss Chamfer Distance Cal time: 817.698 ms

------ START Pytorch3d Chamfer Distance Cal ------
Pytorch3d Chamfer Distance Cal time: 68.256 ms
loss:  tensor(0.1710, device='cuda:0', grad_fn=<AddBackward0>)

------ START mmcv 2D Chamfer Distance Cal ------
loss:  tensor(0.0591, device='cuda:0', grad_fn=<AddBackward0>)
mmcv Chamfer Distance Cal time: 651.510 ms

------ START CUDA Chamfer Distance Cal ------
Chamfer Distance Cal time: 14.308 ms
loss:  tensor(0.1710, device='cuda:0', grad_fn=<AddBackward0>)

"""