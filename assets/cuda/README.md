My CUDA library
---

Faster our code in CUDA.

- chamfer3D: 3D chamfer distance within two point cloud, by Qingwen Zhang involved when she was working on SeFlow.
- mmcv: directly from mmcv, not our code.
- mmdet: only python file, no need to compile
- histlib: from Yancong's [ICP-Flow](https://github.com/yanconglin/ICP-Flow) project.

---

Quick View about CUDA speed on our chamfer3D (Faster 60x than others):

The number of points: (pc0: 88132, pc1: 88101)

| Function | Time (ms) |
| :---: | :---: |
| Faiss | 817.698 |
| CUDA([SCOOP](https://github.com/itailang/SCOOP/tree/master/auxiliary/ChamferDistancePytorch), Batch) | 83.275 |
| Pytorch3D | 68.256 |
| CUDA([SeFlow](https://github.com/KTH-RPL/SeFlow), SharedM) | **14.308** |
| ~~mmcv~~(chamfer2D) | 651.510 |