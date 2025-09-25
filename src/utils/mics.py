import numpy as np
from itertools import accumulate
from collections import namedtuple
from typing import Optional

import torch.nn.init as init
import torch

import pickle, h5py, os

# ref: https://github.com/Lightning-AI/lightning/issues/924#issuecomment-1670917204
def weights_update(model, checkpoint):
    print("Loading pretrained weights from epoch:", checkpoint['epoch']+1)
    model_dict = model.state_dict()

    pretrained_dict = {}
    for k, v in checkpoint['state_dict'].items():
        k = k.replace("model.", "")
        if k in model_dict:
            pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                
def setup_multi_gpu(if_multi_gpu: bool):
    n_gpu = torch.cuda.device_count()
    if n_gpu>1 and if_multi_gpu:
        torch.distributed.init_process_group(backend='nccl')
    else:
        if_multi_gpu = False
        n_gpu = 1
    assert n_gpu, "Can't find any GPU device on this machine."

    return n_gpu

def init_weights(m) -> None:
    """
    Apply the weight initialization to a single layer.
    Use this with your_module.apply(init_weights).
    The single layer is a module that has the weights parameter. This does not yield for all modules.
    :param m: the layer to apply the init to
    :return: None
    """
    if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
        # Note: There is also xavier_normal_ but the paper does not state which one they used.
        torch.nn.init.xavier_uniform_(m.weight)

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
    
def ground_range_filter(av2_sweep, range_lim=50):
    pointcloud = av2_sweep.lidar.as_tensor().numpy()[:,:3]
    ground_mask = ~av2_sweep.is_ground.numpy()
    range_mask = np.where((pointcloud[:,0] > -range_lim) & (pointcloud[:,0] < range_lim) & 
                        (pointcloud[:,1] > -range_lim) & (pointcloud[:,1] < range_lim) & 
                        (pointcloud[:,2] > -range_lim) & (pointcloud[:,2] < range_lim))
    return pointcloud[ground_mask&range_mask]

def ground_range_mask(av2_sweep, range_lim=50):
    pointcloud = av2_sweep.lidar.as_tensor().numpy()[:,:3]
    ground_mask = ~av2_sweep.is_ground.numpy()
    range_mask = ((pointcloud[:,0] > -range_lim) & (pointcloud[:,0] < range_lim) & 
                (pointcloud[:,1] > -range_lim) & (pointcloud[:,1] < range_lim) & 
                (pointcloud[:,2] > -range_lim) & (pointcloud[:,2] < range_lim))
    return ground_mask&range_mask

def ground_range_tmask(pointcloud, ground, range_lim=50):
    pointcloud = pointcloud.numpy()[:,:3]
    ground_mask = ~ground.numpy()
    range_mask = ((pointcloud[:,0] > -range_lim) & (pointcloud[:,0] < range_lim) & 
                (pointcloud[:,1] > -range_lim) & (pointcloud[:,1] < range_lim) & 
                (pointcloud[:,2] > -range_lim) & (pointcloud[:,2] < range_lim))
    return ground_mask&range_mask

# ====> import func through string, ref: https://stackoverflow.com/a/19393328
import importlib
def import_func(path: str):
    function_string = path
    mod_name, func_name = function_string.rsplit('.',1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


# plotting from FastNSF: 

# ANCHOR: visualization as in the paper
DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

def make_colorwheel(transitions: tuple=DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0])
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float]=None,
    background: Optional[str]="bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(f"background should be one the following: {valid_backgrounds}, not {background}.")
    wheel = make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int32)] * (1 - angle_fractional) + wheel[angle_ceil.astype(np.int32)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        'ColorizationArgs', ['move_hue_valid_radius', 'move_hue_oversized_radius', 'invalid_color']
    )
    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)
    def move_hue_on_S_axis(hues, factors):
        return 255. - np.expand_dims(factors, -1) * (255. - hues)
    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float64)
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float64))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)


class HDF5Data:
    def __init__(self, directory, flow_view=False, vis_name=["flow"]):
        '''
        directory: the directory of the dataset
        t_x: how many past frames we want to extract
        '''
        self.flow_view = flow_view
        self.vis_name = vis_name if isinstance(vis_name, list) else [vis_name]
        self.directory = directory
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, index):
        scene_id, timestamp = self.data_index[index]
        # to make sure we have continuous frames for flow view
        if self.flow_view and self.scene_id_bounds[scene_id]["max_index"] == index:
            index = index - 1
            scene_id, timestamp = self.data_index[index]

        key = str(timestamp)
        data_dict = {
            'scene_id': scene_id,
            'timestamp': timestamp,
        }
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            # original data
            data_dict['pc0'] = f[key]['lidar'][:][:,:3]
            data_dict['gm0'] = f[key]['ground_mask'][:]
            data_dict['pose0'] = f[key]['pose'][:]
            for flow_key in self.vis_name + ['dufo', 'label', 'lidar_id', 'lidar_center']:
                if flow_key in f[key]:
                    data_dict[flow_key] = f[key][flow_key][:]

            if self.flow_view:
                next_timestamp = str(self.data_index[index+1][1])
                data_dict['pose1'] = f[next_timestamp]['pose'][:]
                data_dict['pc1'] = f[next_timestamp]['lidar'][:]
                data_dict['gm1'] = f[next_timestamp]['ground_mask'][:]
            elif self.flow_view:
                print(f"[Warning]: No {self.vis_name} in {scene_id} at {timestamp}, check the data.")
        return data_dict
    
from av2.geometry.se3 import SE3
from scipy.spatial.transform import Rotation as R
def transform_to_array(pose):
    pose_se3 = SE3(rotation=pose[:3,:3], translation=pose[:3,3])
    qxyzw = R.from_matrix(pose_se3.rotation).as_quat()
    pose_array = [pose_se3.translation[0], pose_se3.translation[1], pose_se3.translation[2], \
        qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]
    return pose_array

from zipfile import ZipFile
import time, json
def zip_res(res_folder, output_file="av2_submit.zip", leaderboard_version=2, is_supervised=False):
    """
    res_folder: the folder of the output results
    """
    all_scenes = os.listdir(res_folder)
    start_time = time.time()

    if leaderboard_version == 1:
        with ZipFile(output_file, "w") as myzip:
            for scene in all_scenes:
                scene_folder = os.path.join(res_folder, scene)
                all_logs = os.listdir(scene_folder)
                for log in all_logs:
                    if not log.endswith(".feather"):
                        continue
                    file_path = os.path.join(scene, log)
                    myzip.write(os.path.join(res_folder, file_path), arcname=file_path)
    else:
        output_file = output_file.replace(".zip", f"_v{leaderboard_version}.zip") if output_file=="av2_submit.zip" else output_file
        metadata = {"Is Supervised?": is_supervised}
        with ZipFile(output_file, "w") as myzip:
            myzip.writestr("metadata.json", json.dumps(metadata, indent=4))
            for scene in all_scenes:
                scene_folder = os.path.join(res_folder, scene)
                relative_idx = 0
                for log in sorted(os.listdir(scene_folder)):
                    if not log.endswith(".feather"):
                        continue
                    file_path = os.path.join(scene, log)
                    myzip.write(os.path.join(res_folder, file_path), arcname=os.path.join(scene, f"{relative_idx:010d}.feather"))
                    relative_idx += 5
    print(f"Time cost: {time.time()-start_time:.2f}s, check the zip file: {output_file}")
    return output_file