"""
# Created: 2023-11-04 15:52
# Updated: 2024-07-12 23:16
# 
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Jaeyeul Kim (jykim94@dgist.ac.kr)
#
# Change Logs:
# 2024-11-06: Added Data Augmentation transform for RandomHeight, RandomFlip, RandomJitter from DeltaFlow project.
# 2024-07-12: Merged num_frame based on Flow4D model from Jaeyeul Kim.
# 
# Description: Torch dataloader for the dataset we preprocessed.
# 
# This file is part of 
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# 
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import torch, re
from torch.utils.data import Dataset, DataLoader
import h5py, pickle, argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from torchvision import transforms

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils import import_func

def extract_flow_number(key):
    """Get the number at the end of a string, return '0' if none found"""
    digits = re.findall(r'\d+$', key)
    return digits[0] if digits else '0'

def pad_tensors(data_dict, padding_value):
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        for key, tensor_list in data_dict.items()
    }

def collate_fn_pad(batch):
    """Group and pad batch data by key type"""
    batch_size = len(batch)

    point_clouds, labels, poses, flows = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    all_ego_motion_flag = True
    # will process the point clouds, labels, poses, and flows
    for item in batch:
        for key, value in item.items():
            frame_id = key.split('_')[0]  # Extract frame identifier
            ground_mask_key = f"gm{frame_id[2:]}"  # Create ground mask key
            if key.startswith('pc') and ground_mask_key in item:
                masked_value = value[~item[ground_mask_key]]
                if key.endswith("dynamic"):
                    labels[key].append(masked_value)
                else:
                    point_clouds[key].append(masked_value)
            elif key.startswith('flow'):
                flow_id = extract_flow_number(key)
                ground_mask_key = f'gm{flow_id}'
                masked_value = value[~item[ground_mask_key]]
                flows[key].append(masked_value)
            elif key.startswith('pose'):
                poses[key].append(value)
        if 'ego_motion' not in item:
            all_ego_motion_flag = False

    padded_point_clouds = pad_tensors(point_clouds, torch.nan)
    padded_flows = pad_tensors(flows, 0)
    padded_labels = pad_tensors(labels, 0)

    result = {**padded_point_clouds, **padded_labels, **padded_flows}

    # Add poses as lists (no padding needed)
    for key, pose_list in poses.items():
        result[key] = [pose_list[i] for i in range(batch_size)]

    # Add additional fields if present
    if all_ego_motion_flag:
        result['ego_motion'] = [item['ego_motion'] for item in batch]

    # Always include scene_id
    result['scene_id'] = [item['scene_id'] for item in batch]

    return result

# transform, augment
class RandomJitter(object):
    "Randomly add small noise to the point cloud."
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        for key in data_dict.keys():
            if key.startswith("pc") and not key.endswith("dynamic"):
                jitter = np.clip(
                    self.sigma * np.random.randn(data_dict[key].shape[0], 3),
                    -self.clip,
                    self.clip,
                )
                data_dict[key] += jitter
        return data_dict

class RandomFlip(object):
    def __init__(self, p=0.5, verbose=False):
        """p: probability of flipping"""
        self.p = p
        self.verbose = verbose

    def __call__(self, data_dict):
        flip_x = np.random.rand() < self.p
        flip_y = np.random.rand() < self.p

        # If no flip, return directly
        if not (flip_x or flip_y):
            return data_dict
        
        for key in data_dict.keys():
            if (key.startswith("pc") or (key.startswith("flow") and data_dict[key].dtype == np.float32)) and not key.endswith("dynamic"):
                if flip_x:
                    data_dict[key][:, 0] = -data_dict[key][:, 0]
                if flip_y:
                    data_dict[key][:, 1] = -data_dict[key][:, 1]
            if key.startswith("pose"):
                if flip_x:
                    pose = data_dict[key].copy()
                    pose[:, 0] *= -1
                    data_dict[key] = pose
                if flip_y:
                    pose = data_dict[key].copy()
                    pose[:, 1] *= -1
                    data_dict[key] = pose

        if "ego_motion" in data_dict:
            # need recalculate the ego_motion
            data_dict["ego_motion"] = np.linalg.inv(data_dict['pose1']) @ data_dict['pose0']
        if self.verbose:
            print(f"RandomFlip: flip_x={flip_x}, flip_y={flip_y}")
        return data_dict

class RandomHeight(object):
    def __init__(self, p=0.5, verbose=False):
        """p: probability of changing height"""
        self.p = p
        self.verbose = verbose

    def __call__(self, data_dict):
        # NOTE(Qingwen): The reason set -0.5 to 2.0 is because some dataset axis origin is around the ground level. (vehicle base etc.)
        random_height = np.random.uniform(-0.5, 2.0)
        if np.random.rand() < self.p:
            for key in data_dict.keys():
                if key.startswith("pc") and not key.endswith("dynamic"):
                    data_dict[key][:, 2] += random_height
            if self.verbose:
                print(f"RandomHeight: {random_height}")
        return data_dict

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, data_dict):
        for key in data_dict.keys():
            # skip the scene_id, timestamp, eval_flag to tensor conversion
            if key in ['scene_id', 'timestamp', 'eval_flag']:
                continue
            elif isinstance(data_dict[key], np.ndarray):
                data_dict[key] = torch.tensor(data_dict[key])
            else:
                print(f"Warning: {key} is not a numpy array. Type: {type(data_dict[key])}")
        return data_dict

class LayeredFrame:
    """One frame of a scene: its group in the base h5 file, and its groups in the layer h5 files.

    frame['dufo'] reads the key from the layer that has it, otherwise from the base file.
    So a dataset that still has everything inside the base file reads the same as before.
    """
    def __init__(self, base, layers):
        self.base, self.layers = base, layers

    def __contains__(self, name):
        return name in self.base or any(name in layer for layer in self.layers)

    def __getitem__(self, name):
        found = [layer for layer in self.layers if name in layer]
        if len(found) > 1:
            raise KeyError(f"'{name}' is in more than one layer: {[layer.file.filename for layer in found]}")
        if not found and name not in self.base:
            raise KeyError(f"'{name}' is not in {self.base.file.filename}, nor in its layers: {[layer.file.filename for layer in self.layers]}")
        return found[0][name] if found else self.base[name]

class LayeredScene:
    """The h5 files of one scene, opened read-only: {directory}/{scene_id}.h5 and the same file name in every layer folder.

    scene[timestamp] is the LayeredFrame of that frame. A layer may only have some of the scenes or frames.
    """
    def __init__(self, directory, layer_dirs, scene_id):
        self.base = h5py.File(os.path.join(directory, f'{scene_id}.h5'), 'r')
        layer_files = [os.path.join(layer_dir, f'{scene_id}.h5') for layer_dir in layer_dirs]
        self.layers = [h5py.File(path, 'r') for path in layer_files if os.path.exists(path)]

    def __getitem__(self, timestamp):
        return LayeredFrame(self.base[timestamp], [layer[timestamp] for layer in self.layers if timestamp in layer])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for file in [self.base] + self.layers:
            file.close()

class HDF5Dataset(Dataset):
    def __init__(self, directory, \
                transform=None, n_frames=2, ssl_label=None, loss_frames=None, \
                eval = False, future_frames=1, pad_frames=True, \
                vis_name='', index_flow=False, layers=None):
        '''
        Args:
            directory: the directory of the dataset, the folder should contain some .h5 file and index_total.pkl.

            Following are optional:
            * transform: for data augmentation, default is None.
            * n_frames: the number of frames we use, default is 2: current (pc0), next (pc1); if it's more than 2, then it read the history from current.
            * ssl_label: if attr, it will read the dynamic cluster label. Otherwise, no dynamic cluster label in data dict.
            * loss_frames: {past: P, future: F}, frames around pc0 that the self-supervised loss uses: pch1..pchP and pc1..pcF with their labels.
                           The model input still follows n_frames. Default None: only the frames from n_frames.
            * eval: if True, use the eval index (only used it for leaderboard evaluation)
            * future_frames: the number of frames after pc0 to read, default is 1: pc1. 3 means pc1, pc2, pc3.
            * pad_frames: what to do with a history frame before the start of the scene. True: repeat the first frame of the scene,
                          so the number of frames is always the same. False: leave it out. A frame after the end of the scene is always left out.
            * vis_name: the data of the visualization, default is ''.
            * index_flow: if True, use the flow index for training or visualization.
            * layers: names of the layers to read on top of the base .h5 files.
                      A layer is the folder {directory}/layers/{name}/ with one {scene_id}.h5 per scene: data that is not
                      part of the dataset itself (labels, predictions, ...), so it can be made again or removed without touching the base files.
                      Default None: 'seflow_labels' (where process.py writes the labels) if that folder exists.
        '''
        super(HDF5Dataset, self).__init__()
        self.directory = directory
        if layers is None:
            layers = ['seflow_labels'] if os.path.isdir(os.path.join(directory, 'layers', 'seflow_labels')) else []
        self.layer_dirs = [os.path.join(directory, 'layers', name) for name in layers]
        for layer_dir in self.layer_dirs:
            if not os.path.isdir(layer_dir):
                raise FileNotFoundError(f"Layer folder {layer_dir} does not exist.")
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print(f"----[Debug] Loading data with num_frames={n_frames}, ssl_label={ssl_label}, eval={eval}")
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        self.eval_index = False
        self.ssl_label = import_func(f"src.autolabel.{ssl_label}") if ssl_label is not None else None
        self.history_frames = n_frames - 2
        self.future_frames, self.pad_frames = future_frames, pad_frames
        self.loss_past, self.loss_future = 0, 1
        if loss_frames is not None:
            if ssl_label is None or eval:
                raise ValueError("loss_frames is only for self-supervised training: it needs ssl_label and eval=False.")
            self.loss_past, self.loss_future = int(loss_frames['past']), int(loss_frames['future'])
        self.vis_name = vis_name if isinstance(vis_name, list) else [vis_name]
        # a saved flow may be in its own layer: {directory}/layers/pred/{vis_name}/
        pred_dirs = [os.path.join(directory, 'layers', 'pred', name) for name in self.vis_name if name]
        self.layer_dirs += [pred_dir for pred_dir in pred_dirs if os.path.isdir(pred_dir)]
        self.transform = transform

        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if not os.path.exists(eval_index_file):
                print(f"Warning: No {eval_index_file} file found! We will try {'index_flow.pkl'}")
                eval_index_file = os.path.join(self.directory, 'index_flow.pkl')
                if not os.path.exists(eval_index_file):
                    raise Exception(f"No any eval index file found! Please check {self.directory}")
            
            self.eval_index = eval
            with open(eval_index_file, 'rb') as f:
                self.eval_data_index = pickle.load(f)

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp, "max_timestamp": timestamp,
                    "min_index": idx, "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx
        
        # for some dataset that annotated HZ is different.... like truckscene and nuscene etc.
        self.train_index = None
        if (not eval and ssl_label is None and transform is not None) or index_flow: # transform indicates whether we are in training mode.
            # check if train seq all have gt.
            one_scene_id = list(self.scene_id_bounds.keys())[0]
            check_flow_exist = True
            with h5py.File(os.path.join(self.directory, f'{one_scene_id}.h5'), 'r') as f:
                for i in range(self.scene_id_bounds[one_scene_id]["min_index"], self.scene_id_bounds[one_scene_id]["max_index"]):
                        scene_id, timestamp = self.data_index[i]
                        key = str(timestamp)
                        if 'flow' not in f[key]:
                            check_flow_exist = False
                            break
            if not check_flow_exist:
                print(f"----- [Warning]: Not all frames have flow data, we will instead use the index_flow.pkl to train.")
                self.train_index = pickle.load(open(os.path.join(self.directory, 'index_flow.pkl'), 'rb'))
                
    def __len__(self):
        # return 100 # for testing
        if self.eval_index:
            return len(self.eval_data_index)
        elif not self.eval_index and self.train_index is not None:
            return len(self.train_index)
        return len(self.data_index)
    
    def valid_index(self, index_):
        """
        Check if the index is valid for the current mode and satisfy the constraints.
        """
        eval_flag = False
        if self.eval_index:
            eval_index_ = index_
            scene_id, timestamp = self.eval_data_index[eval_index_]
            index_ = self.data_index.index([scene_id, timestamp])
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            if index_ >= max_idx:
                _, index_ = self.valid_index(eval_index_ - 1)
            eval_flag = True
        elif self.train_index is not None:
            train_index_ = index_
            scene_id, timestamp = self.train_index[train_index_]
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            index_ = self.data_index.index([scene_id, timestamp])
            if index_ >= max_idx:
                _, index_ = self.valid_index(train_index_ - 1)
        else:
            scene_id, timestamp = self.data_index[index_]
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            min_idx = self.scene_id_bounds[scene_id]["min_index"]

            # keep every frame we read (model input and loss frames) inside the scene
            max_valid_index_for_flow = max_idx - max(self.future_frames, self.loss_future)
            min_valid_index_for_flow = min_idx + max(self.history_frames, self.loss_past)
            if not self.pad_frames: # the frames outside the scene are left out in __getitem__, only pc1 has to be there
                max_valid_index_for_flow, min_valid_index_for_flow = max_idx - 1, min_idx
            index_ = max(min_valid_index_for_flow, min(max_valid_index_for_flow, index_))
        return eval_flag, index_
    
    def __getitem__(self, index_):
        eval_flag, index_ = self.valid_index(index_)
        scene_id, timestamp = self.data_index[index_]

        key = str(timestamp)
        data_dict = {
            'scene_id': scene_id,
            'timestamp': timestamp,
            'eval_flag': eval_flag
        }
        with LayeredScene(self.directory, self.layer_dirs, scene_id) as f:
            # original data
            data_dict['pc0'] = f[key]['lidar'][:][:,:3]
            data_dict['gm0'] = f[key]['ground_mask'][:]
            data_dict['pose0'] = f[key]['pose'][:]
            if self.ssl_label is not None:
                data_dict['pc0_dynamic'] = self.ssl_label(f[key])

            if self.history_frames >= 0: 
                next_timestamp = str(self.data_index[index_ + 1][1])
                data_dict['pose1'] = f[next_timestamp]['pose'][:]
                data_dict['pc1'] = f[next_timestamp]['lidar'][:][:,:3]
                data_dict['gm1'] = f[next_timestamp]['ground_mask'][:]
                if self.ssl_label is not None:
                    data_dict['pc1_dynamic'] = self.ssl_label(f[next_timestamp])
                
                past_frames = []
                for i in range(1, max(self.history_frames, self.loss_past) + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        if not self.pad_frames:
                            break
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = f[past_timestamp]['lidar'][:][:,:3]
                    past_gm = f[past_timestamp]['ground_mask'][:]
                    past_pose = f[past_timestamp]['pose'][:]

                    past_frames.append((past_pc, past_gm, past_pose))
                    if self.ssl_label is not None: # every history frame is supervised in the multi-frame loss
                        data_dict[f'pch{i}_dynamic'] = self.ssl_label(f[past_timestamp])

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    data_dict[f'pch{i+1}'] = past_pc
                    data_dict[f'gmh{i+1}'] = past_gm
                    data_dict[f'poseh{i+1}'] = past_pose

                # frames after pc1: pc2, pc3, ... as model input (future_frames) or only for the loss (loss_frames)
                for i in range(2, max(self.future_frames, self.loss_future) + 1):
                    if index_ + i > self.scene_id_bounds[scene_id]["max_index"]:
                        break # end of the scene, only happens in eval: the training index is kept inside in valid_index()
                    future_timestamp = str(self.data_index[index_ + i][1])
                    data_dict[f'pose{i}'] = f[future_timestamp]['pose'][:]
                    data_dict[f'pc{i}'] = f[future_timestamp]['lidar'][:][:,:3]
                    data_dict[f'gm{i}'] = f[future_timestamp]['ground_mask'][:]
                    if self.ssl_label is not None:
                        data_dict[f'pc{i}_dynamic'] = self.ssl_label(f[future_timestamp])

            for data_key in self.vis_name + ['ego_motion', 'lidar_dt', 'lidar_center',
                             # ground truth information:
                             'flow', 'flow_is_valid', 'flow_category_indices', 'flow_instance_id', 'dufo']:
                if data_key in f[key]:
                    data_dict[data_key] = f[key][data_key][:]

            if self.eval_index:
                # looks like v2 not follow the same rule as v1 with eval_mask provided
                if 'eval_mask' in f[key]:
                    raw_eval = f[key]['eval_mask'][:]
                    raw_ground = f[key]['ground_mask'][:]
                    # NOTE(Qingwen): performance might be changed for av2 since some eval_mask provided by av2 didn't remove ground points.
                    data_dict['eval_mask'] = (raw_eval.reshape(-1).astype(bool) & (~raw_ground.reshape(-1).astype(bool)))
                elif 'ground_mask' in f[key]:
                    data_dict['eval_mask'] = ~f[key]['ground_mask'][:]
                else:
                    data_dict['eval_mask'] = np.ones_like(data_dict['pc0'][:, 0], dtype=np.bool_)
                    
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader test")
    parser.add_argument('--data_mode', '-m', type=str, default='train', metavar='N', help='Dataset mode.')
    parser.add_argument('--data_dir', '-d', type=str, default='/home/kin/data/av2/h5py_v2/sensor', metavar='N', help='preprocess data path.')
    options = parser.parse_args()

    # testing eval mode
    dataset = HDF5Dataset(directory = options.data_dir+"/"+options.data_mode, eval = False,
                          transform = transforms.Compose([RandomHeight(), RandomFlip(), RandomJitter(), ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn_pad)
    for data in tqdm(dataloader, ncols=80, desc="read data mode"):
        res_dict = data
        # print(res_dict['pc0'].shape)
        # break