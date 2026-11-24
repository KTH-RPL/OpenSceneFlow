Reproducing the optimization-based methods
---

How our re-implementations of [Floxels](#floxels) and [EulerFlow](#eulerflow) are set up, where they differ from the papers and the original code, and what we get on the Argoverse 2 (AV2) val set.
Both run without training, directly through `eval.py` / `save.py`, see [README](../../README.md#optimization-based-unsupervised-methods).

## Floxels

[Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](https://arxiv.org/abs/2503.04718) (CVPR 2025). Code: [src/models/floxels.py](../../src/models/floxels.py), settings: [conf/model/floxels.yaml](../../conf/model/floxels.yaml).
We thank the Floxels authors for the discussion and their [pull request](https://github.com/Kin-Zhang/OpenSceneFlow/pull/2) to our re-implementation.

```bash
pip install --no-build-isolation --no-cache-dir FastGeodis==1.0.5   # the distance transform needs it
python eval.py model=floxels   # 5 frames: t-2 ... t+2
```

Settings:
- The flow of pc0 is a voxel grid (0.5 m) inside ±80 m (z ±3 m), the flow of a point is the trilinear interpolation of the grid.
- 5 frames, pch2 ... pc2. The data term of frame k (distance transform with a 1/6 m cell, truncated at 5 m) is weighted with 1/k².
- Cluster term: weight 0.6, DBSCAN (eps 0.5 m, 4 points) on pc0. Flow norm term: weight from 0.1 to 0.01 over the first 100 iterations.
- 500 iterations of Adam, learning rate 0.05, halved at iteration 10 and 20. Early stopping as in the paper: stop when the loss did not get 0.01 lower in the last 250 iterations. The flow with the lowest loss is returned.

Where we differ:
- Early stopping: with a short patience (20 iterations, min_delta 1e-4) many frames stop before the loss gets below that of the zero flow, and the result is the zero flow (about 45% of the frames on 15 AV2 val scenes). The values of the paper (250, 0.01) do not have this problem.
- DBSCAN runs on the CPU (scikit-learn): about 16% of the runtime.
- At the start and the end of a scene, the frames that do not exist are left out (`pad_frames: False`), the window is not shortened on the other side.

Results on the whole AV2 val set (150 scenes), 5 frames, about 4.4 s per frame on an RTX 3090:

| mean Dynamic | mean Static | Three-way EPE | EPE FD | EPE BS | EPE FS | CAR | OTHER_VEHICLES | PEDESTRIAN | WHEELED_VRU |
|---|---|---|---|---|---|---|---|---|---|
| 0.191 | 0.0227 | 0.0384 | 0.0561 | 0.0300 | 0.0291 | 0.097 | 0.281 | 0.260 | 0.126 |

The paper reports a mean Dynamic of 0.217 on AV2 val. The AV2 val ground truth of this repo was corrected in [#5](https://github.com/KTH-RPL/OpenSceneFlow/pull/5), so the two numbers are not computed on exactly the same ground truth.

## EulerFlow

[Neural Eulerian Scene Flow Fields](https://arxiv.org/abs/2410.02031) (ICLR 2025). Code: [src/models/eulerflow.py](../../src/models/eulerflow.py), settings: [conf/model/eulerflow.yaml](../../conf/model/eulerflow.yaml).
It follows the released code [SceneFlowZoo](https://github.com/kylevedder/SceneFlowZoo) (MIT license), `models/mini_batch_optimization/eulerflow.py` with `configs/eulerflow/argo/noncausal_minibatched/val.py`.

```bash
python eval.py model=eulerflow
python eval.py model=eulerflow procs_per_gpu=2   # two processes on each GPU, about 1.6x faster on an RTX 3080
```

One network is fitted on all frames of a scene and saved in `model.target.ckpt_dir`, so a stopped run goes on from there.
The loss is the same as in SceneFlowZoo (checked to 7 digits on the same input). Our default fit is shorter:

| | SceneFlowZoo | ours (default) |
|---|---|---|
| epochs | 1000 | 200, early stop when the loss did not get 0.001 lower in 150 epochs |
| learning rate | 8e-5 | 8e-4 for 150 epochs, then 8e-5 |
| points of a frame moved in a step | all | 8000 random ones, compared with all points of the neighbor frames |
| point cloud range | ±48 m | ±80 m (outside: the ego-motion flow) |
| frames of a scene | the first 150 | all |
| time per AV2 scene, RTX 3090 | 5.8 - 7.5 h | 19 - 21 min |

Other settings: an MLP of 8 layers with width 128 (the paper has its best result with 18 layers), 5 frames in one step, multi-step loss up to 3 frames.
The cycle term has weight 1: SceneFlowZoo writes `* 0.01` for it, but its cost classes drop the factor when they sum, so 1 is what it runs with.

Results on AV2 val:
- 15 scenes of the val set: mean Dynamic 0.172 (SceneFlowZoo reports 0.159 on the whole val set).
- Two demo scenes, our default fit vs 1000 epochs of SceneFlowZoo: 0.118 vs 0.180 and 0.245 vs 0.290.
- Range: on the 15 val scenes, the dynamic error at 50-75 m is 0.74 with ±48 m and about 0.1 with ±80 m; within 35 m both are about the same.
