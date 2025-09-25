import numpy as np
import pickle, h5py, os, time
from pathlib import Path
from tqdm import tqdm

from typing import Tuple, cast

def check_h5py_file_exists(h5py_file: Path, timestamps: list, verbose: bool = False) -> bool:
    if not h5py_file.exists():
        return False
    
    log_id = h5py_file.name.split(".")[0]
    try:
        with h5py.File(h5py_file, 'r') as f:
            for timestamp in timestamps:
                if str(timestamp) not in f.keys():
                    # delete the file if the timestamp is not in the file
                    # and it will reprocess this scene file
                    if verbose:
                        print(f"\n--- WARNING [data]: {log_id} has no {timestamp}, will be deleted the scene h5py.")
                    os.remove(h5py_file)
                    return False
    except Exception as e:
        if verbose:
            print(f"\n--- WARNING [data]: {log_id} has error: {e}, will be deleted the scene h5py.")
        os.remove(h5py_file)
        return False
    if verbose:
        print(f'\n--- INFO [data]: {log_id} has been processed with total {len(timestamps)} timestamps.')
    return True

def create_reading_index(data_dir: Path, flow_inside_check=False):
    pkl_file_name = 'index_total.pkl' if not flow_inside_check else 'index_flow.pkl'
    start_time = time.time()
    data_index = []
    for file_name in tqdm(os.listdir(data_dir), ncols=100):
        if not file_name.endswith(".h5"):
            continue
        scene_id = file_name.split(".")[0]
        timestamps = []
        with h5py.File(data_dir/file_name, 'r') as f:
            if flow_inside_check:
                for key in f.keys():
                    if 'flow' in f[key]:
                        # print(f"Found flow in {scene_id} at {key}")
                        timestamps.append(key)
            else:
                timestamps.extend(f.keys())
        timestamps.sort(key=lambda x: int(x)) # make sure the timestamps are in order
        for timestamp in timestamps:
            data_index.append([scene_id, timestamp])

    with open(data_dir/pkl_file_name, 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create {pkl_file_name} index Successfully, cost: {time.time() - start_time:.2f} s")

def find_closest_integer_in_ref_arr(query_int, ref_arr) -> Tuple[int, int, int]:
    """Find the closest integer to any integer inside a reference array, and the corresponding difference.

    In our use case, the query integer represents a nanosecond-discretized timestamp, and the
    reference array represents a numpy array of nanosecond-discretized timestamps.

    Instead of sorting the whole array of timestamp differences, we just
    take the minimum value (to speed up this function).

    Args:
        query_int: query integer,
        ref_arr: Numpy array of integers

    Returns:
        integer, representing the closest integer found in a reference array to a query
        integer, representing the integer difference between the match and query integers
    """
    closest_ind = np.argmin(np.absolute(ref_arr - query_int))
    closest_int = cast(
        int, ref_arr[closest_ind]
    )  # mypy does not understand numpy arrays
    int_diff = np.absolute(query_int - closest_int)
    return closest_ind, closest_int, int_diff

class SE2:

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Initialize.
        Args:
            rotation: np.ndarray of shape (2,2).
            translation: np.ndarray of shape (2,1).
        Raises:
            ValueError: if rotation or translation do not have the required shapes.
        """
        assert rotation.shape == (2, 2)
        assert translation.shape == (2, )
        self.rotation = rotation
        self.translation = translation
        self.transform_matrix = np.eye(3)
        self.transform_matrix[:2, :2] = self.rotation
        self.transform_matrix[:2, 2] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(2) transformation to point_cloud.
        Args:
            point_cloud: np.ndarray of shape (N, 2).
        Returns:
            transformed_point_cloud: np.ndarray of shape (N, 2).
        Raises:
            ValueError: if point_cloud does not have the required shape.
        """
        assert point_cloud.ndim == 2
        assert point_cloud.shape[1] == 2
        num_points = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_points, 1))])
        transformed_point_cloud = homogeneous_pts.dot(self.transform_matrix.T)
        return transformed_point_cloud[:, :2]

    def inverse(self) -> "SE2":
        """Return the inverse of the current SE2 transformation.
        For example, if the current object represents target_SE2_src, we will return instead src_SE2_target.
        Returns:
            inverse of this SE2 transformation.
        """
        return SE2(rotation=self.rotation.T,
                   translation=self.rotation.T.dot(-self.translation))

    def inverse_transform_point_cloud(self,
                                      point_cloud: np.ndarray) -> np.ndarray:
        """Transform the point_cloud by the inverse of this SE2.
        Args:
            point_cloud: Numpy array of shape (N,2).
        Returns:
            point_cloud transformed by the inverse of this SE2.
        """
        return self.inverse().transform_point_cloud(point_cloud)

    def compose(self, right_se2: "SE2") -> "SE2":
        """Multiply this SE2 from right by right_se2 and return the composed transformation.
        Args:
            right_se2: SE2 object to multiply this object by from right.
        Returns:
            The composed transformation.
        """
        chained_transform_matrix = self.transform_matrix.dot(
            right_se2.transform_matrix)
        chained_se2 = SE2(
            rotation=chained_transform_matrix[:2, :2],
            translation=chained_transform_matrix[:2, 2],
        )
        return chained_se2
    

## ====> nuScenes to Argoverse Mapping
NusNamMap = {
    'noise': 'NONE',
    'animal': 'ANIMAL',
    'human.pedestrian.adult': 'PEDESTRIAN',
    'human.pedestrian.child': 'PEDESTRIAN',
    'human.pedestrian.construction_worker': 'PEDESTRIAN',
    'human.pedestrian.personal_mobility': 'PEDESTRIAN',
    'human.pedestrian.police_officer': 'PEDESTRIAN',
    'human.pedestrian.stroller': 'STROLLER',
    'human.pedestrian.wheelchair': 'WHEELCHAIR',
    'movable_object.barrier': 'NONE',
    'movable_object.debris': 'NONE',
    'movable_object.pushable_pullable': 'NONE',
    'movable_object.trafficcone': 'CONSTRUCTION_CONE',
    'static_object.bicycle_rack': 'NONE',
    'vehicle.bicycle': 'BICYCLE',
    'vehicle.bus.bendy': 'ARTICULATED_BUS',
    'vehicle.bus.rigid': 'BUS',
    'vehicle.car': 'REGULAR_VEHICLE',
    'vehicle.construction': 'LARGE_VEHICLE',
    'vehicle.emergency.ambulance': 'LARGE_VEHICLE',
    'vehicle.emergency.police': 'REGULAR_VEHICLE',
    'vehicle.motorcycle': 'MOTORCYCLE',
    'vehicle.trailer': 'VEHICULAR_TRAILER',
    'vehicle.truck': 'TRUCK',
    'flat.driveable_surface': 'NONE',
    'flat.other': 'NONE',
    'flat.sidewalk': 'NONE',
    'flat.terrain': 'NONE',
    'static.manmade': 'NONE',
    'static.other': 'NONE',
    'static.vegetation': 'NONE',
    'vehicle.ego': 'NONE'
}

## ====> MAN to Argoverse Mapping
ManNamMap = {
    "animal": 'NONE',
    "human.pedestrian.adult": 'PEDESTRIAN',
    "human.pedestrian.child": 'PEDESTRIAN',
    "human.pedestrian.construction_worker": 'PEDESTRIAN',
    "human.pedestrian.personal_mobility": 'PEDESTRIAN',
    "human.pedestrian.police_officer": 'PEDESTRIAN',
    "human.pedestrian.stroller": 'STROLLER',
    "human.pedestrian.wheelchair": 'WHEELCHAIR',
    "movable_object.barrier": 'NONE',
    "movable_object.debris": 'NONE',
    "movable_object.pushable_pullable": 'NONE',
    "movable_object.trafficcone": 'CONSTRUCTION_CONE',
    "static_object.bicycle_rack": 'NONE',
    "static_object.traffic_sign": 'SIGN',
    "vehicle.bicycle": 'BICYCLE',
    "vehicle.bus.bendy": 'ARTICULATED_BUS',
    "vehicle.bus.rigid": 'BUS',
    "vehicle.car": 'REGULAR_VEHICLE',
    "vehicle.construction": 'LARGE_VEHICLE',
    "vehicle.emergency.ambulance": 'LARGE_VEHICLE',
    "vehicle.emergency.police": 'REGULAR_VEHICLE',
    "vehicle.motorcycle": 'MOTORCYCLE',
    "vehicle.trailer": 'VEHICULAR_TRAILER',
    "vehicle.truck": 'TRUCK',
    "vehicle.train": 'NONE',
    "vehicle.other": 'NONE',
    "vehicle.ego_trailer": 'NONE',
}