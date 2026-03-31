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
from tqdm import tqdm
import numpy as np
from torchvision import transforms

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils import import_func

def extract_flow_number(key):
    digits = re.findall(r'\d+$', key)
    if digits:
        return digits[0]
    return '0'

# FIXME(Qingwen 2025-08-20): update more pretty here afterward!
def collate_fn_pad(batch):
    batch_size_ = len(batch)
    pcs_after_mask_ground, poses_dict, flows_after_mask_ground = {}, {}, {}
    
    for i in range(batch_size_):
        single_data = batch[i]
        for key in single_data.keys():
            if key.startswith('pc') and f'gm{key[2:]}' in single_data and not key.endswith("dynamic"):
                gm_key = f'gm{key[2:]}'
                if key not in pcs_after_mask_ground:
                    pcs_after_mask_ground[key] = []
                pcs_after_mask_ground[key].append(single_data[key][~single_data[gm_key]])
            elif key.startswith('flow'):
                id_flow = extract_flow_number(key)
                gm_key = f'gm{id_flow}'
                if key not in flows_after_mask_ground:
                    flows_after_mask_ground[key] = []
                flows_after_mask_ground[key].append(single_data[key][~single_data[gm_key]])
            elif key.startswith('pose'):
                if key not in poses_dict:
                    poses_dict[key] = []
                poses_dict[key].append(single_data[key])

    for key in pcs_after_mask_ground:
        pcs_after_mask_ground[key] = torch.nn.utils.rnn.pad_sequence(pcs_after_mask_ground[key], batch_first=True, padding_value=torch.nan)
    for key in flows_after_mask_ground:
        flows_after_mask_ground[key] = torch.nn.utils.rnn.pad_sequence(flows_after_mask_ground[key], batch_first=True)

    # Prepare the result dictionary
    res_dict = {key: pcs_after_mask_ground[key] for key in pcs_after_mask_ground}
    # ground truth information:
    res_dict.update({key: flows_after_mask_ground[key] for key in flows_after_mask_ground})
    res_dict.update({key: [poses_dict[key][i] for i in range(batch_size_)] for key in poses_dict})

    for flow_key in flows_after_mask_ground:
        flows_after_mask_ground[flow_key] = torch.nn.utils.rnn.pad_sequence(flows_after_mask_ground[flow_key], batch_first=True, padding_value=0)
        res_dict[flow_key] = flows_after_mask_ground[flow_key]

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(batch_size_)]
        
    if 'pc0_dynamic' in batch[0]:
        pc0_dynamic_after_mask_ground, pc1_dynamic_after_mask_ground= [], []
        for i in range(batch_size_):
            pc0_dynamic_after_mask_ground.append(batch[i]['pc0_dynamic'][~batch[i]['gm0']])
            pc1_dynamic_after_mask_ground.append(batch[i]['pc1_dynamic'][~batch[i]['gm1']])
        pc0_dynamic_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_dynamic_after_mask_ground, batch_first=True, padding_value=0)
        pc1_dynamic_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_dynamic_after_mask_ground, batch_first=True, padding_value=0)
        res_dict['pc0_dynamic'] = pc0_dynamic_after_mask_ground
        res_dict['pc1_dynamic'] = pc1_dynamic_after_mask_ground
    if 'pch1_dynamic' in batch[0]:
        pch_dynamic_after_mask_ground = [batch[i]['pch1_dynamic'][~batch[i]['gmh1']] for i in range(batch_size_)]
        pch_dynamic_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pch_dynamic_after_mask_ground, batch_first=True, padding_value=0)
        res_dict['pch1_dynamic'] = pch_dynamic_after_mask_ground
    
    # save the scene_id also...
    res_dict['scene_id'] = [batch[i]['scene_id'] for i in range(batch_size_)]

    return res_dict

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

class HDF5Dataset(Dataset):
    def __init__(self, directory, \
                transform=None, n_frames=2, ssl_label=None, \
                eval = False, leaderboard_version=1, \
                vis_name='', index_flow=False):
        '''
        Args:
            directory: the directory of the dataset, the folder should contain some .h5 file and index_total.pkl.

            Following are optional:
            * transform: for data augmentation, default is None.
            * n_frames: the number of frames we use, default is 2: current (pc0), next (pc1); if it's more than 2, then it read the history from current.
            * ssl_label: if attr, it will read the dynamic cluster label. Otherwise, no dynamic cluster label in data dict.
            * eval: if True, use the eval index (only used it for leaderboard evaluation)
            * leaderboard_version: 1st or 2nd, default is 1. If '2', we will use the index_eval_v2.pkl from assets/docs.
            * vis_name: the data of the visualization, default is ''.
            * index_flow: if True, use the flow index for training or visualization.
        '''
        super(HDF5Dataset, self).__init__()
        self.directory = directory
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print(f"----[Debug] Loading data with num_frames={n_frames}, ssl_label={ssl_label}, eval={eval}, leaderboard_version={leaderboard_version}")
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        self.eval_index = False
        self.ssl_label = import_func(f"src.autolabel.{ssl_label}") if ssl_label is not None else None
        self.history_frames = n_frames - 2
        self.vis_name = vis_name if isinstance(vis_name, list) else [vis_name]
        self.transform = transform

        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if leaderboard_version == 2:
                print("Using index to leaderboard version 2!!")
                eval_index_file = os.path.join(BASE_DIR, 'assets/docs/index_eval_v2.pkl')

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

            max_valid_index_for_flow = max_idx - 1
            min_valid_index_for_flow = min_idx + self.history_frames
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
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
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
                for i in range(1, self.history_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = f[past_timestamp]['lidar'][:][:,:3]
                    past_gm = f[past_timestamp]['ground_mask'][:]
                    past_pose = f[past_timestamp]['pose'][:]

                    past_frames.append((past_pc, past_gm, past_pose))
                    if i == 1 and self.ssl_label is not None: # only for history 1: t-1
                        # data_dict['pch1_dynamic'] = f[past_timestamp]['label'][:].astype('int16')
                        data_dict['pch1_dynamic'] = self.ssl_label(f[past_timestamp])

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    data_dict[f'pch{i+1}'] = past_pc
                    data_dict[f'gmh{i+1}'] = past_gm
                    data_dict[f'poseh{i+1}'] = past_pose

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