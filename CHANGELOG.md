# Changelog

Notable changes of OpenSceneFlow and the public scene flow leaderboard, newest first.

## 2026-11-05

- **Leaderboard**: The leaderboard v3 has moved to [sceneflow.kth-rpl.se](https://sceneflow.kth-rpl.se/leaderboard). The old domain `sceneflow.argoverse.org` can no longer upload zip files, and single-request `curl` uploads are not supported anymore. Please submit with `sceneflow_submit.sh`, see [README](README.md#submit-result-to-public-leaderboard).
- **Speed**: The decoders `SparseGRUHead` (DeltaFlow) and `Point_head` (DeltaFlow default & Flow4D) now gather the point voxel features directly from the sparse tensor instead of `.dense()`. Same results with the released checkpoints. On an RTX 3090: DeltaFlow inference 34 → 26 ms, training peak memory 4.4 → 1.2 GB (batch size 2); Flow4D inference 42 → 35 ms, training peak memory 7.6 → 5.1 GB.
- **Runtime**: New option `timer_sync` (default `False`) for train/eval/save. With `timer_sync=True`, the timers synchronize CUDA at every start and stop, so the timetable reports the real GPU runtime of each part; it also prints the timetable at the end of every training epoch.

## 2026-04-10

- **Leaderboard**: We've updated the leaderboard to [version 3](https://sceneflow.kth-rpl.se/leaderboard) with more datasets and longer range. You can specify the `leaderboard_version` in the config to select which version you want to submit and report the item you want in the paper.

## 2024-09-11

- **Leaderboard**: The leaderboard results in DeFlow & SeFlow main papers are [version 1](https://eval.ai/web/challenges/challenge-page/2010/evaluation), as [version 2](https://eval.ai/web/challenges/challenge-page/2210/overview) was updated after DeFlow & SeFlow.
