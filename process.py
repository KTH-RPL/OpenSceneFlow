"""
# Created: 2023-11-04 15:55
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of SeFlow (https://github.com/KTH-RPL/SeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: run dufomap on the dataset we preprocessed for afterward ssl training.
#              it's only needed for ssl train but not inference.
#              Goal to segment dynamic and static point roughly.
"""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import fire, time, h5py, os
from hdbscan import HDBSCAN

from src.utils.mics import HDF5Data, transform_to_array
from dufomap import dufomap

MIN_AXIS_RANGE = 2 # HARD CODED: remove ego vehicle points
MAX_AXIS_RANGE = 50 # HARD CODED: remove far away points

def run_cluster(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/train",
    scene_range: list = [0, 1],
    interval: int = 1, # useless here, just for the same interface args
    overwrite: bool = False,
):
    data_path = Path(data_dir)
    dataset = HDF5Data(data_path)
    all_scene_ids = list(dataset.scene_id_bounds.keys())
    for scene_in_data_index, scene_id in enumerate(all_scene_ids):
        start_time = time.time()
        # NOTE (Qingwen): so the scene id range is [start, end)
        if scene_range[0]!= -1 and scene_range[-1]!= -1 and (scene_in_data_index < scene_range[0] or scene_in_data_index >= scene_range[1]):
            continue
        bounds = dataset.scene_id_bounds[scene_id]
        flag_exist_label = True
        with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
            for ii in range(bounds["min_index"], bounds["max_index"]+1):
                key = str(dataset[ii]['timestamp'])
                if 'label' not in f[key]:
                    flag_exist_label = False
                    break
        if flag_exist_label and not overwrite:
            print(f"==> Scene {scene_id} has plus label, skip.")
            continue
        
        hdb = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=0.7)
        for i in tqdm(range(bounds["min_index"], bounds["max_index"]+1), desc=f"Start Plus Cluster: {scene_in_data_index}/{len(all_scene_ids)}", ncols=80):
            data = dataset[i]
            pc0 = data['pc0'][:,:3]
            cluster_label = np.zeros(pc0.shape[0], dtype= np.int16)

            if "dufo_label" not in data:
                print(f"Warning: {scene_id} {data['timestamp']} has no dufo_label, will be skipped. Better to rerun dufomap again in this scene.")
                continue
            elif data["dufo_label"].sum() < 20:
                print(f"Warning: {scene_id} {data['timestamp']} has no dynamic points, will be skipped. Better to check this scene.")
            else:
                hdb.fit(pc0[data["dufo_label"]==1])
                # NOTE(Qingwen): since -1 will be assigned if no cluster. We set it to 0.
                cluster_label[data["dufo_label"]==1] = hdb.labels_ + 1 

            # save labels
            timestamp = data['timestamp']
            key = str(timestamp)
            with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
                if 'label' in f[key]:
                    # print(f"Warning: {scene_id} {timestamp} has label, will be overwritten.")
                    del f[key]['label']
                f[key].create_dataset('label', data=np.array(cluster_label).astype(np.int16))
        print(f"==> Scene {scene_id} finished, used: {(time.time() - start_time)/60:.2f} mins")
    print(f"Data inside {str(data_path)} finished. Check the result with tools/visulization.py if you want to visualize them.")

def run_dufo(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/train",
    scene_range: list = [0, 1],
    interval: int = 1, # interval frames to run dufomap
    overwrite: bool = False,
):
    data_path = Path(data_dir)
    dataset = HDF5Data(data_path)
    all_scene_ids = list(dataset.scene_id_bounds.keys())
    for scene_in_data_index, scene_id in enumerate(all_scene_ids):
        start_time = time.time()
        # NOTE (Qingwen): so the scene id range is [start, end)
        if scene_range[0]!= -1 and scene_range[-1]!= -1 and (scene_in_data_index < scene_range[0] or scene_in_data_index >= scene_range[1]):
            continue
        bounds = dataset.scene_id_bounds[scene_id]
        flag_has_dufo_label = True
        with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
            for ii in range(bounds["min_index"], bounds["max_index"]+1):
                key = str(dataset[ii]['timestamp'])
                if "dufo_label" not in f[key]:
                    flag_has_dufo_label = False
                    break
        if flag_has_dufo_label and not overwrite:
            print(f"==> Scene {scene_id} has dufo_label, skip.")
            continue

        mydufo = dufomap(0.2, 0.2, 1, num_threads=12) # resolution, d_s, d_p, hit_extension
        mydufo.setCluster(0, 20, 0.2) # depth=0, min_points=20, max_dist=0.2

        print(f"==> Scene {scene_id} start, data path: {data_path}")
        for i in tqdm(range(bounds["min_index"], bounds["max_index"]+1), desc=f"Dufo run: {scene_in_data_index}/{len(all_scene_ids)}", ncols=80):
            if interval != 1 and i % interval != 0 and (i + interval//2 < bounds["max_index"] or i - interval//2 > bounds["min_index"]):
                continue
            data = dataset[i]
            assert data['scene_id'] == scene_id, f"Check the data, scene_id {scene_id} is not consistent in {i}th data in {scene_in_data_index}th scene."
            # HARD CODED: remove points outside the range
            norm_pc0 = np.linalg.norm(data['pc0'][:, :3], axis=1)
            range_mask = (
                    (norm_pc0>MIN_AXIS_RANGE) & 
                    (norm_pc0<MAX_AXIS_RANGE)
            )
            pose_array = transform_to_array(data['pose0'])
            mydufo.run(data['pc0'][range_mask], pose_array, cloud_transform = True)

        # finished integrate, start segment, needed since we have map.label inside dufo
        mydufo.oncePropagateCluster(if_cluster = True, if_propagate=True)
        for i in tqdm(range(bounds["min_index"], bounds["max_index"]+1), desc=f"Start Segment: {scene_in_data_index}/{len(all_scene_ids)}", ncols=80):
            data = dataset[i]
            pc0 = data['pc0']
            gm0 = data['gm0']
            pose_array = transform_to_array(data['pose0'])
            dufo_label = np.array(mydufo.segment(pc0, pose_array, cloud_transform = True))
            dufo_labels = np.zeros(pc0.shape[0], dtype= np.uint8)
            dufo_labels[~gm0] = dufo_label[~gm0]

            # save labels
            timestamp = data['timestamp']
            key = str(timestamp)
            with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
                if "dufo_label" in f[key]:
                    # print(f"Warning: {scene_id} {timestamp} has label, will be overwritten.")
                    del f[key]["dufo_label"]
                f[key].create_dataset("dufo_label", data=np.array(dufo_labels).astype(np.uint8))
        print(f"==> Scene {scene_id} finished, used: {(time.time() - start_time)/60:.2f} mins")
    print(f"Data inside {str(data_path)} finished. Check the result with vis() function if you want to visualize them.")

if __name__ == '__main__':
    start_time = time.time()
    # step 1: run dufomap
    fire.Fire(run_dufo)
    # step 2: run cluster on dufolabel
    fire.Fire(run_cluster)

    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")