"""
# Created: 2025-11-21 15:13
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description:
    Create evaluation index pickle file from the total index pickle file.
    - need have enough non-ground points (as some of waymo frames have data quality issues)
    - sample every 5 frames for evaluation (followed the leaderboard setting) it can also save 5x validation time for optimization-based methods also.
"""

import os, fire, pickle, time
import h5py, torch
from tqdm import tqdm

def create_evalpkl(
    data_dir: str = "/home/kin/data/waymo/valid",
    interval: int = 5,
):
    with open(os.path.join(data_dir, "index_total.pkl"), 'rb') as f:
        total_index = pickle.load(f)

    scene_id_bounds = {}
    for idx, (scene_id, timestamp) in enumerate(total_index):
        if scene_id not in scene_id_bounds:
            scene_id_bounds[scene_id] = {
                "min_timestamp": timestamp, "max_timestamp": timestamp,
                "min_index": idx, "max_index": idx
            }
        else:
            bounds = scene_id_bounds[scene_id]
            if timestamp < bounds["min_timestamp"]:
                bounds["min_timestamp"] = timestamp
                bounds["min_index"] = idx
            if timestamp > bounds["max_timestamp"]:
                bounds["max_timestamp"] = timestamp
                bounds["max_index"] = idx

    # split the index by 5 - 5 frame, start with the fifth frame
    eval_data_index = []
    for scene_id, bounds in tqdm(scene_id_bounds.items(), desc="Creating eval index", total=len(scene_id_bounds), dynamic_ncols=True):
        with h5py.File(os.path.join(data_dir, f'{scene_id}.h5'), 'r') as f:
            for idx in range(bounds["min_index"] + interval*2, bounds["max_index"] - interval*2, interval):
                scene_id, timestamp = total_index[idx]
                key = str(timestamp)
                pc = torch.tensor(f[key]['lidar'][:][:,:3])
                gm = torch.tensor(f[key]['ground_mask'][:])
                if pc[~gm].shape[0] < 10000:
                    continue
                eval_data_index.append(total_index[idx])
    
    # print(f"Demo: {eval_data_index[:10]}")
    print(f"Total {len(eval_data_index)} frames for evaluation in {data_dir}.")
    with open(os.path.join(data_dir, "index_eval.pkl"), 'wb') as f:
        pickle.dump(eval_data_index, f)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(create_evalpkl)
    print(f"Create reading index Successfully, cost: {time.time() - start_time:.2f} s")