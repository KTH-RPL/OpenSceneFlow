# Changelog

Notable changes of OpenSceneFlow and the public scene flow leaderboard, newest first.

## 2026-11-22

- **Data**: Labels and saved flows are not written into the dataset `.h5` files anymore, they go to layers next to them: `process.py` writes the self-supervised labels to `{data_dir}/layers/ssl_labels/`, `save.py` and `eval.py save_res=True` write the flow to `{data_dir}/layers/pred/{res_name}/`. The dataloader and the visualization tools read them from there. So the dataset files no longer grow with every saved result, and labels or results can be shared, uploaded or removed (`rm -r`) on their own. Files that still have labels or flows inside read the same as before, `tools/h5_split_layers.py` moves them out.
- **Method**: New optimization-based method [EulerFlow](https://arxiv.org/abs/2410.02031) (ICLR 2025): `python eval.py model=eulerflow`. Our default fit (200 epochs: a high learning rate on random points first, then a low one) is about 20x faster than the fit of SceneFlowZoo with the same performance, see [reproduce details](assets/docs/reproduce.md#eulerflow).
- **Method**: New optimization-based method [Floxels](https://arxiv.org/abs/2503.04718) (CVPR 2025), our re-implementation: `python eval.py model=floxels`, see [reproduce details](assets/docs/reproduce.md#floxels).
- **Environment**: The default environment is now Python 3.10 + torch 2.6.0 + CUDA 11.8 (`environment.yaml`, or `requirements.txt` for pip), see [README](README.md#environment-setup).
- **Runtime**: `eval.py` and `save.py` run every model (feed-forward and optimization-based) through `src/runner.py`, Lightning is only used for training.
- **Runtime**: New option `procs_per_gpu` for eval/save: several processes on one GPU, each on its own scenes, for methods where one process cannot keep the GPU busy (EulerFlow on an RTX 3080: about 1.6x).

## 2026-11-05

- **Leaderboard**: The leaderboard v3 has moved to [sceneflow.kth-rpl.se](https://sceneflow.kth-rpl.se/leaderboard). The old domain `sceneflow.argoverse.org` can no longer upload zip files, and single-request `curl` uploads are not supported anymore. Please submit with `sceneflow_submit.sh`, see [README](README.md#submit-result-to-public-leaderboard).
- **Speed**: The decoders `SparseGRUHead` (DeltaFlow) and `Point_head` (DeltaFlow default & Flow4D) now gather the point voxel features directly from the sparse tensor instead of `.dense()`. Same results with the released checkpoints. On an RTX 3090: DeltaFlow inference 34 → 26 ms, training peak memory 4.4 → 1.2 GB (batch size 2); Flow4D inference 42 → 35 ms, training peak memory 7.6 → 5.1 GB.
- **Runtime**: New option `timer_sync` (default `False`) for train/eval/save. With `timer_sync=True`, the timers synchronize CUDA at every start and stop, so the timetable reports the real GPU runtime of each part; it also prints the timetable at the end of every training epoch.

## 2026-04-10

- **Leaderboard**: We've updated the leaderboard to [version 3](https://sceneflow.kth-rpl.se/leaderboard) with more datasets and longer range. You can specify the `leaderboard_version` in the config to select which version you want to submit and report the item you want in the paper.

## 2024-09-11

- **Leaderboard**: The leaderboard results in DeFlow & SeFlow main papers are [version 1](https://eval.ai/web/challenges/challenge-page/2010/evaluation), as [version 2](https://eval.ai/web/challenges/challenge-page/2210/overview) was updated after DeFlow & SeFlow.
