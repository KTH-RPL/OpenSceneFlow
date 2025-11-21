"""
# Created: 2023-12-31 22:19
# LastEdit: 2024-01-12 18:46
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description:
#   Quick Read the keys in an h5 file, print out their shapes and data types etc.

# Example Running:
    python tools/readh5.py --scene_path /home/kin/data/av2/h5py/sensor/test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68.h5
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import fire, time, h5py

def readh5key(
    scene_path: str = "/home/kin/data/av2/h5py/sensor/test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68.h5"
):
    with h5py.File(scene_path, 'r') as f:
        for cnt, k in enumerate(f.keys()):
            if cnt % 2 == 1:
                continue
            print(f"id: {cnt}; Key (TimeStamp): {k}")
            for sub_k in f[k].keys():
                print(f"  Sub-Key: {sub_k}, Shape: {f[k][sub_k].shape}, Dtype: {f[k][sub_k].dtype}")
            if cnt >= 10:
                break
        print(f"\nTotal {len(f.keys())} timestamps in the file.")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(readh5key)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")