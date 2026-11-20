"""
# Created: 2026-11-20 21:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), AI Assistant (Claude Code)
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.
#
# Description: Split dataset .h5 files that have the labels and the saved flows inside, into base files + layers:
#
#   {data_dir}/{scene}.h5                         what dataprocess/ wrote: lidar, pose, ground_mask, flow, ...
#   {data_dir}/layers/seflow_labels/{scene}.h5    dufo, cluster, dufocluster, nnd (process.py)
#   {data_dir}/layers/pred/{name}/{scene}.h5      every other key: a flow written by save.py / eval.py save_res
#
# A saved flow is as big as the lidar points of the frame, so a few of them make the dataset several times bigger.
# In layers they can be removed (rm -r layers/pred/{name}) or made again without touching the base files.
#
# Three steps, nothing is changed unless asked:
#   python tools/h5_split_layers.py --data_dir /path/to/val            # dry run: what would move, and how much space it is
#   python tools/h5_split_layers.py --data_dir /path/to/val --copy     # copy into layers and compare every array. Base files untouched.
#   python tools/h5_split_layers.py --data_dir /path/to/val --strip    # rewrite the base files without the keys that are identical in a layer
#
# Options: --scenes "id1,id2" only these scenes (to run in parallel on a cluster), --fp16 store the saved flows as float16 in the layer.
# NOTE: --strip rewrites one base file at a time into a temporary file next to it, so it needs the free space of one file.
#       Do not run it while something else reads or writes the dataset.
"""

import os, json, shutil, time
from collections import defaultdict
import numpy as np
import h5py, fire
from tabulate import tabulate

# what dataprocess/extract_*.py writes, these stay in the base file
BASE_KEYS = ['lidar', 'lidar_id', 'lidar_dt', 'lidar_center', 'pose', 'ego_motion', 'ground_mask',
             'flow', 'flow_is_valid', 'flow_category_indices', 'flow_instance_id', 'eval_mask']
# what process.py writes
LABEL_KEYS = ['dufo', 'cluster', 'dufocluster', 'nnd']

def layer_of(key):
    """The layer a key of the dataset file belongs to, None if it stays in the base file."""
    if key in BASE_KEYS:
        return None
    if key in LABEL_KEYS:
        return 'seflow_labels'
    return f'pred/{key}' # anything else was saved by save.py / eval.py save_res under its res_name

def frames_of(f):
    """The frames (groups named by timestamp) of an opened dataset file."""
    return [ts for ts in f if isinstance(f[ts], h5py.Group)]

def same(base, layer):
    """Does the dataset in a layer hold the data of the dataset in the base file? Compared in the dtype of the layer (float16 with --fp16)."""
    if base.shape != layer.shape:
        return False
    return np.array_equal(base[()].astype(layer.dtype), layer[()], equal_nan=layer.dtype.kind == 'f')

def stored_bytes(path):
    """Bytes on disk of every key of one dataset file, summed over its frames: {key: bytes}."""
    size = defaultdict(int)
    with h5py.File(path, 'r') as f:
        for ts in frames_of(f):
            for key, dataset in f[ts].items():
                size[key] += dataset.id.get_storage_size()
    return size

def copy_scene(path, data_dir, fp16):
    """Copy the non-base keys of one scene into its layer files. A layer file that is already there is left as it is."""
    scene, written = os.path.basename(path), []
    with h5py.File(path, 'r') as f:
        for layer in sorted({layer_of(key) for ts in frames_of(f) for key in f[ts]} - {None}):
            out_path = os.path.join(data_dir, 'layers', layer, scene)
            if os.path.exists(out_path):
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            keys = set()
            with h5py.File(out_path + '.tmp', 'w') as out:
                for ts in frames_of(f):
                    for key in [key for key in f[ts] if layer_of(key) == layer]:
                        group = out.require_group(ts)
                        dataset = f[ts][key]
                        if fp16 and layer.startswith('pred/') and dataset.dtype == np.float32:
                            group.create_dataset(key, data=dataset[()].astype(np.float16),
                                                 compression=dataset.compression, compression_opts=dataset.compression_opts)
                        else:
                            f.copy(dataset, group, key)
                        keys.add(key)

            # the layer file only gets its final name after every array in it is compared with the base file
            with h5py.File(out_path + '.tmp', 'r') as out:
                wrong = [(ts, key) for ts in frames_of(f) for key in f[ts] if layer_of(key) == layer and not same(f[ts][key], out[ts][key])]
            if wrong:
                os.remove(out_path + '.tmp')
                raise RuntimeError(f"{scene}: the copy of {wrong[0]} in layer '{layer}' is not the same as in the base file, nothing was kept.")
            os.replace(out_path + '.tmp', out_path)
            written.append(layer)

            meta_path = os.path.join(data_dir, 'layers', layer, '_meta.json')
            if not os.path.exists(meta_path):
                with open(meta_path, 'w') as meta:
                    json.dump({'keys': sorted(keys), 'made_by': 'tools/h5_split_layers.py: moved out of the dataset .h5 files, how it was generated is unknown',
                               'float16': bool(fp16 and layer.startswith('pred/')), 'time': time.strftime('%Y-%m-%d %H:%M')}, meta, indent=1)
    return written

def strip_scene(path, data_dir):
    """Rewrite one base file without the keys that are identical in a layer file. Returns the bytes it got smaller."""
    scene, layer_files = os.path.basename(path), {}
    with h5py.File(path, 'r') as f:
        drop, keep_differ = set(), set()
        for ts in frames_of(f):
            for key in f[ts]:
                layer = layer_of(key)
                if layer is None:
                    continue
                if layer not in layer_files:
                    layer_path = os.path.join(data_dir, 'layers', layer, scene)
                    layer_files[layer] = h5py.File(layer_path, 'r') if os.path.exists(layer_path) else None
                layer_file = layer_files[layer]
                if layer_file is not None and ts in layer_file and key in layer_file[ts] and same(f[ts][key], layer_file[ts][key]):
                    drop.add((ts, key))
                else:
                    keep_differ.add(key)
        for layer_file in layer_files.values():
            if layer_file is not None:
                layer_file.close()
        if keep_differ:
            print(f"  {scene}: {sorted(keep_differ)} stay in the base file: not (or not identical) in a layer. Run --copy first"
                  f" (after removing the old layer file of this scene if the data was made again).")
        if not drop:
            return 0

        kept_bytes = sum(f[ts][key].id.get_storage_size() for ts in frames_of(f) for key in f[ts] if (ts, key) not in drop)
        if shutil.disk_usage(data_dir).free < kept_bytes * 1.05:
            raise RuntimeError(f"{scene}: not enough free space in {data_dir} for the temporary file ({kept_bytes / 2**30:.2f} GB).")

        frames = set(frames_of(f))
        with h5py.File(path + '.tmp', 'w') as out:
            out.attrs.update(f.attrs)
            for name in f:
                if name not in frames: # not a frame, keep it as it is
                    f.copy(f[name], out, name)
                    continue
                group = out.create_group(name)
                group.attrs.update(f[name].attrs)
                for key in f[name]:
                    if (name, key) not in drop:
                        f.copy(f[name][key], group, key)

        # the new base file only replaces the old one after every array that stays is compared
        with h5py.File(path + '.tmp', 'r') as out:
            wrong = [(ts, key) for ts in frames_of(f) for key in f[ts] if (ts, key) not in drop and not same(f[ts][key], out[ts][key])]
        if wrong:
            os.remove(path + '.tmp')
            raise RuntimeError(f"{scene}: {wrong[0]} in the new base file is not the same as before, the base file was not changed.")
    old_size = os.path.getsize(path)
    os.replace(path + '.tmp', path)
    return old_size - os.path.getsize(path)

def main(data_dir: str, copy: bool = False, strip: bool = False, fp16: bool = False, scenes = None):
    paths = sorted(os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.h5'))
    if scenes is not None:
        scenes = scenes if isinstance(scenes, (list, tuple)) else str(scenes).split(',')
        paths = [path for path in paths if os.path.basename(path)[:-3] in scenes]
    print(f"{len(paths)} scenes in {data_dir}")

    if not copy and not strip:
        total, file_bytes = defaultdict(int), 0
        for path in paths:
            file_bytes += os.path.getsize(path)
            for key, size in stored_bytes(path).items():
                total[key] += size
        rows = [[key, layer_of(key) or '(base file)', f"{size / 2**30:.2f}", f"{100 * size / max(file_bytes, 1):.1f}%"]
                for key, size in sorted(total.items(), key=lambda item: -item[1])]
        print(tabulate(rows, headers=['key', 'goes to layers/', 'GB', 'of the files'], tablefmt='github'))
        movable = sum(size for key, size in total.items() if layer_of(key) is not None)
        print(f"\nFiles now: {file_bytes / 2**30:.2f} GB. In layers: {movable / 2**30:.2f} GB, base files after --strip: about {(file_bytes - movable) / 2**30:.2f} GB.")
        print("Nothing was changed. Next: --copy (base files untouched), then --strip.")
        return

    if copy:
        for i, path in enumerate(paths):
            written = copy_scene(path, data_dir, fp16)
            print(f"[copy {i+1}/{len(paths)}] {os.path.basename(path)}: {written if written else 'nothing new to copy'}")
    if strip:
        saved = 0
        for i, path in enumerate(paths):
            saved_scene = strip_scene(path, data_dir)
            saved += saved_scene
            print(f"[strip {i+1}/{len(paths)}] {os.path.basename(path)}: {saved_scene / 2**30:.2f} GB smaller")
        print(f"The base files are {saved / 2**30:.2f} GB smaller.")

if __name__ == '__main__':
    fire.Fire(main)
